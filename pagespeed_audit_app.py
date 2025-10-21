import os
import re
import time
import json
import queue
import fnmatch
import urllib.parse as urlparse
from typing import Dict, List, Optional, Set
import concurrent.futures
import threading
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st

PAGESPEED_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

SUMMARY_METRIC_IDS = {
    "Performance Score": ("categories.performance.score", None),
    "Largest Contentful Paint (LCP)": ("audits.largest-contentful-paint.numericValue", "ms"),
    "Speed Index": ("audits.speed-index.numericValue", "ms"),
    "Time to Interactive (TTI)": ("audits.interactive.numericValue", "ms"),
    "Cumulative Layout Shift (CLS)": ("audits.cumulative-layout-shift.numericValue", None),
    "First Contentful Paint (FCP)": ("audits.first-contentful-paint.numericValue", "ms"),
    "Total Blocking Time (TBT)": ("audits.total-blocking-time.numericValue", "ms"),
}

HUMAN_LABELS = {
    "uses-text-compression": "Enable text compression",
    "render-blocking-resources": "Eliminate render-blocking resources",
    "server-response-time": "Reduce initial server response time",
    "unused-css-rules": "Reduce unused CSS",
    "unminified-css": "Minify CSS",
    "unminified-javascript": "Minify JavaScript",
    "unused-javascript": "Reduce unused JavaScript",
    "total-byte-weight": "Avoid enormous network payloads",
    "uses-optimized-images": "Properly size images",
    "modern-image-formats": "Serve images in next-gen formats",
    "efficient-animated-content": "Use video formats for animated content",
    "offscreen-images": "Defer offscreen images",
    "diagnostics": "Diagnostics",
    "dom-size": "Avoid an excessive DOM size",
    "critical-request-chains": "Avoid chaining critical requests",
    "max-potential-fid": "Max Potential First Input Delay",
    "layout-shifts": "Avoid large layout shifts",
    "unsized-images": "Image elements without explicit size",
    "font-display": "Use font-display",
    "legacy-javascript": "Legacy JavaScript",
    "mainthread-work-breakdown": "Minimize main-thread work",
    "resource-summary": "Resources summary"
}

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PageSpeed-Audit-Tool/1.0; +https://example.com)"}

# ---------------------- NEW: Exclusion patterns ----------------------
DEFAULT_EXCLUDE_PATTERNS = [
    "/wp-content/*",
    "/wp-includes/*",
    "/wp-json/*",
    "/wp-admin/*",
    "/?s=*",
    "/search*",
    "/feed*",
    "/tag/*",
    "/category/*",
    "/author/*",
    "/cart*",
    "/checkout*",
    "/my-account*",
    "/login*",
    "/logout*",
    "/register*",
    "/sitemap*",
    "/robots.txt",
    "*.xml",
    "*.pdf",
    "*.doc",
    "*.docx",
    "*.zip",
    "*.rar",
    "*.7z",
    "*.mp4",
    "*.mp3",
    "*.webm",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.gif",
    "*.webp",
    "*.svg",
    "*.ico",
    "*.css",
    "*.js",
]


def _deep_get(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def _format_metric(value, unit: Optional[str]):
    if value is None:
        return None
    if unit == "ms":
        return float(value)
    if isinstance(value, (int, float)) and 0 <= value <= 1:
        return round(value * 100, 0)
    return value


def _impact_label_from_opportunity(audit: dict) -> str:
    details = audit.get("details", {}) or {}
    overall_ms = details.get("overallSavingsMs") or 0
    overall_bytes = details.get("overallSavingsBytes") or 0
    score = audit.get("score")
    if overall_ms >= 800 or overall_bytes >= 150_000 or (isinstance(score, (int, float)) and score <= 0.3):
        return "High"
    if overall_ms >= 200 or overall_bytes >= 50_000 or (isinstance(score, (int, float)) and score <= 0.6):
        return "Medium"
    return "Low"


def _audit_title(audit_id: str, audit: dict) -> str:
    return audit.get("title") or HUMAN_LABELS.get(audit_id) or audit_id


def _norm_url(u: str) -> str:
    return u.strip().split("#")[0] if u else u


def is_internal_link(base_netloc: str, href: str) -> bool:
    try:
        p = urlparse.urlparse(href)
        if not p.netloc:
            return True
        return p.netloc == base_netloc
    except Exception:
        return False


def extract_links(html: str, base_url: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)
    base = urlparse.urlparse(base_url)
    out = []
    for a in anchors:
        href = a["href"]
        if href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"):
            continue
        if href.startswith("#"):
            continue
        if not is_internal_link(base.netloc, href):
            continue
        abs_url = urlparse.urljoin(base_url, href)
        out.append(_norm_url(abs_url))
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            uniq.append(u)
        seen.add(u)
    return uniq


@st.cache_data(show_spinner=False)
def run_pagespeed(url: str, strategy: str, api_key: str) -> dict:
    params = [
        ("url", url),
        ("strategy", strategy),
        ("key", api_key),
        ("category", "performance"),
        ("category", "accessibility"),
        ("category", "best-practices"),
        ("category", "seo"),
    ]
    resp = requests.get(PAGESPEED_ENDPOINT, params=params, headers=DEFAULT_HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ---------------------- NEW: Exclusion helpers ----------------------

def _url_path_query(u: str) -> str:
    """Return lowercased path+query for matching against patterns."""
    p = urlparse.urlparse(u)
    pq = p.path or "/"
    if p.query:
        pq += "?" + p.query
    return pq.lower()


def _should_exclude(url: str, patterns: List[str]) -> bool:
    pq = _url_path_query(url)
    for raw in patterns or []:
        pat = (raw or "").strip()
        if not pat:
            continue
        # Exact match when no wildcard
        if "*" not in pat:
            # Support absolute or relative exacts
            if pat.startswith("http://") or pat.startswith("https://"):
                if _norm_url(url).lower() == _norm_url(pat).lower():
                    return True
            else:
                if not pat.startswith("/"):
                    pat = "/" + pat
                if pq == pat.lower():
                    return True
        else:
            # Wildcard match with fnmatch on path+query for relative patterns; full URL for absolute
            if pat.startswith("http://") or pat.startswith("https://"):
                if fnmatch.fnmatch(_norm_url(url).lower(), pat.lower()):
                    return True
            else:
                if not pat.startswith("/") and not pat.startswith("*"):
                    pat = "/" + pat
                if fnmatch.fnmatch(pq, pat.lower()):
                    return True
    return False


def _get_links(url, base_netloc, seen, lock, delay_sec, exclude_patterns: List[str]):
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
        r.raise_for_status()
        links = extract_links(r.text, url)
        res = []
        for href in links:
            if href in seen:
                continue
            if not is_internal_link(base_netloc, href):
                continue
            if _should_exclude(href, exclude_patterns):
                continue
            with lock:
                if href not in seen:
                    seen.add(href)
                    res.append(href)
            if delay_sec > 0:
                time.sleep(delay_sec)
        return res
    except Exception:
        return []


def crawl_site_multithread_interruptible(start_url, max_pages, delay_sec, workers, exclude_patterns: List[str]):
    start_url = _norm_url(start_url)
    base = urlparse.urlparse(start_url)
    seen: Set[str] = set([start_url])
    q = queue.Queue()
    q.put(start_url)
    results = [start_url]
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while not q.empty() and len(results) < max_pages and st.session_state.crawling:
            batch = []
            while not q.empty() and len(batch) < workers and len(results) < max_pages:
                batch.append(q.get())
            future_to_url = {executor.submit(_get_links, url, base.netloc, seen, lock, delay_sec, exclude_patterns): url for url in batch}
            for future in concurrent.futures.as_completed(future_to_url):
                if not st.session_state.crawling:
                    break
                out_links = future.result()
                for l in out_links:
                    if len(results) < max_pages:
                        results.append(l)
                        q.put(l)
    return results[:max_pages]


def extract_summary(lhr: dict) -> Dict[str, Optional[float]]:
    summary = {}
    for label, (path, unit) in SUMMARY_METRIC_IDS.items():
        if path == "categories.performance.score":
            val = _deep_get(lhr, "lighthouseResult.categories.performance.score")
        else:
            val = _deep_get(lhr, f"lighthouseResult.{path}")
        summary[label] = _format_metric(val, unit)
    return summary


def _list_audits(lhr: dict) -> Dict[str, dict]:
    return _deep_get(lhr, "lighthouseResult.audits") or {}


def _device_from_lhr(lhr: dict) -> str:
    form_factor = _deep_get(lhr, "lighthouseResult.configSettings.formFactor")
    if form_factor in ("mobile", "desktop"):
        return form_factor
    return "mobile"


def get_auditid_to_category(lhr):
    catmap = {}
    cats = _deep_get(lhr, "lighthouseResult.categories") or {}
    for catkey in ["performance", "accessibility", "best-practices", "seo"]:
        c = cats.get(catkey)
        if c and "auditRefs" in c:
            for a in c["auditRefs"]:
                aid = a.get("id")
                if aid:
                    catmap[aid] = c.get("title", catkey.title())
    return catmap


def extract_issues_table(lhr: dict) -> pd.DataFrame:
    audits = _list_audits(lhr)
    device = _device_from_lhr(lhr)
    auditid2cat = get_auditid_to_category(lhr)
    rows = []
    for audit_id, audit in audits.items():
        details = audit.get("details") or {}
        items = details.get("items") or []
        category = None
        if details.get("type") == "opportunity":
            category = "Opportunities"
        elif audit.get("scoreDisplayMode") in ("informative", "notApplicable") or details.get("type") in ("table", "criticalrequestchain", "list"):
            category = "Diagnostics"
        else:
            continue
        title = _audit_title(audit_id, audit)
        impact = _impact_label_from_opportunity(audit) if category == "Opportunities" else "Info"
        category_group = auditid2cat.get(audit_id, "Other")
        if items:
            for it in items:
                url = it.get("url")
                node = it.get("node", {})
                node_label = node.get("nodeLabel") or ""
                source = url or node_label or ""
                val = (
                    it.get("wastedMs")
                    or it.get("wastedBytes")
                    or details.get("overallSavingsMs")
                    or details.get("overallSavingsBytes")
                    or audit.get("displayValue")
                )
                rows.append({
                    "Category Group": category_group,
                    "Issue / Audit": title,
                    "Value": val,
                    "Device": device.capitalize(),
                    "Category": category,
                    "Impact": impact,
                    "URL/Source": source
                })
        else:
            val = audit.get("displayValue") or details.get("overallSavingsMs") or details.get("overallSavingsBytes")
            rows.append({
                "Category Group": category_group,
                "Issue / Audit": title,
                "Value": val,
                "Device": device.capitalize(),
                "Category": category,
                "Impact": impact,
                "URL/Source": ""
            })
    if not rows:
        return pd.DataFrame(columns=["Category Group", "Issue / Audit", "Value", "Device", "Category", "Impact", "URL/Source"])
    df = pd.DataFrame(rows)
    cat_order = {"Opportunities": 0, "Diagnostics": 1}
    imp_order = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}
    df["cat_sort"] = df["Category"].map(cat_order).fillna(99)
    df["imp_sort"] = df["Impact"].map(imp_order).fillna(99)
    def _to_num(v):
        try:
            return float(str(v).replace(",", "").replace("ms", "").strip())
        except Exception:
            return None
    df["val_num"] = df["Value"].apply(_to_num)
    df = df.sort_values(by=["cat_sort", "imp_sort", "val_num"], ascending=[True, True, False]).drop(columns=["cat_sort", "imp_sort", "val_num"])
    return df


def run_pagespeed_safe(u, device, api_key, backup_key=None):
    """Run PageSpeed with main key and fallback to backup if available."""
    try:
        data = run_pagespeed(u, strategy=device, api_key=api_key)
        return u, data, "primary"
    except Exception as e:
        err_msg = str(e)
        if backup_key:
            try:
                data = run_pagespeed(u, strategy=device, api_key=backup_key)
                return u, data, "backup"
            except Exception as e2:
                return u, {"error": f"Primary+Backup failed: {err_msg} | {e2}"}, "failed"
        return u, {"error": f"Primary failed: {err_msg}"}, "failed"



def extract_overview_scores(lhr: dict):
    cats = _deep_get(lhr, "lighthouseResult.categories") or {}
    out = {}
    for cat in ["performance", "accessibility", "best-practices", "seo"]:
        obj = cats.get(cat)
        if obj:
            score = round((obj.get('score', 0.0) or 0.0) * 100, 0)
            out[obj.get('title', cat.title())] = score
        else:
            out[cat.title()] = None
    return out


def score_color(score):
    if score is None:
        return "#999"
    if score >= 90:
        return "#3CBC64"
    if score >= 50:
        return "#FFF176"
    return "#F44336"

# Page config
st.set_page_config(page_title="SiteSpeed Auditor", layout="wide")
# ---------------------- NEW: Hide Streamlit deploy button ----------------------
st.markdown("""
<style>
  [data-testid='stDeployButton'] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.title("🔎 PageSpeed Auditor")

if "crawling" not in st.session_state:
    st.session_state.crawling = False

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Google PageSpeed API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password", disabled=st.session_state.get("running", False) or st.session_state.get("crawling", False))
    backup_api_key = st.text_input(
        "Backup API Key (optional)",
        value=os.getenv("GOOGLE_API_KEY_BACKUP", ""),
        type="password",
        disabled=st.session_state.get("running", False) or st.session_state.get("crawling", False),
        help="Used automatically if the main API key hits quota limits or fails."
    )
    device_options = [
        "Mobile (recommended for SEO)",
        "Desktop",
        "Both (not recommended)"
    ]
    device = st.selectbox("Device strategy", device_options, index=0, disabled=st.session_state.get("running", False) or st.session_state.get("crawling", False))
    max_pages = st.number_input("Max crawl pages", min_value=1, max_value=2000, value=250, step=1, disabled=st.session_state.get("running", False) or st.session_state.get("crawling", False))
    crawl_delay = st.number_input("Crawl delay (seconds)", min_value=0.0, max_value=5.0, value=0.0, step=0.1, disabled=st.session_state.get("running", False) or st.session_state.get("crawling", False))
    crawl_workers = st.number_input("Fetch workers (parallel threads)", min_value=1, max_value=15, value=10, step=1, disabled=st.session_state.get("running", False) or st.session_state.get("crawling", False))
    audit_workers = st.number_input("Audit workers (parallel threads)", min_value=1, max_value=15, value=10, step=1, disabled=st.session_state.get("running", False) or st.session_state.get("crawling", False))

    st.subheader("Audit Mode")
    mode = st.radio("Mode", ["Crawl whole site & measure PageSpeed", "Enter list of URLs to audit (one per line)"], index=0, disabled=st.session_state.get("running", False) or st.session_state.get("crawling", False))
    
    exclude_text_default = "\n".join(DEFAULT_EXCLUDE_PATTERNS)
    exclude_text = st.text_area(
        "Exclude URL patterns",
        value=st.session_state.get("exclude_text", exclude_text_default),
        height=120,
        disabled=st.session_state.get("running", False) or st.session_state.get("crawling", False),
        help=(
            "Enter one exclude pattern per line. "
            "Patterns without '*' must match exactly. Use '*' as a wildcard. "
            "Patterns are matched against the URL path + query (case-insensitive). "
            "Absolute URLs are allowed. Leading '/' is optional for relative patterns. "
            "Examples: /wp-content/*, *.pdf, /login"
        ),
        placeholder="/wp-content/*\n*.pdf\n/login"
    )
    st.session_state.exclude_text = exclude_text
    exclude_patterns = [ln.strip() for ln in exclude_text.splitlines() if ln.strip()]

    st.markdown("---")
    st.subheader("Recent audits (last 10)")
    history = st.session_state.get('audits_history', [])[-10:]
    if history:
        history_labels = [f"{rec['timestamp']} [{rec['device']}] {rec['base_url']}" for rec in history]
        sel_idx = st.selectbox("Select audit", options=list(range(len(history_labels))), format_func=lambda idx: history_labels[idx], key="history_picklist")
        chosen_record = history[sel_idx]
        if st.button("Load audit", key="history_load"):
            st.session_state.urls_to_measure = chosen_record["urls"]
            st.session_state.results = chosen_record["results"]
            st.session_state.idx = 0
            st.success(f"Loaded audit: {chosen_record['base_url']} ({chosen_record['timestamp']})")


for key in ("urls_to_measure", "results", "running", "audits_history", "idx"):
    if key not in st.session_state:
        st.session_state[key] = [] if key in ("urls_to_measure", "audits_history") else ({} if key == "results" else 0)

urls_to_measure = st.session_state.urls_to_measure
results = st.session_state.results

with st.expander("Step 1: Enter or crawl list of URLs", expanded=True):
    if mode == "Crawl whole site & measure PageSpeed":
        start_url = st.text_input("Enter start URL", placeholder="https://example.com/", disabled=st.session_state.crawling or st.session_state.running)
        crawl_col, stop_col = st.columns([1,1])
        with crawl_col:
            crawl_btn = st.button("Fetch pages", disabled=st.session_state.crawling or st.session_state.running)
        with stop_col:
            stop_crawl_btn = st.button("Cancel", disabled=not st.session_state.crawling)
        if crawl_btn and not st.session_state.crawling and not st.session_state.running:
            if not start_url:
                st.warning("Please enter a website URL.")
            else:
                st.session_state.crawling = True
                st.rerun()
        if stop_crawl_btn and st.session_state.crawling:
            st.session_state.crawling = False
        if st.session_state.crawling:
            with st.spinner("Fetching..."):
                urls = crawl_site_multithread_interruptible(
                    start_url,
                    max_pages=int(max_pages),
                    delay_sec=float(crawl_delay),
                    workers=int(crawl_workers),
                    exclude_patterns=exclude_patterns,
                )
                st.session_state.urls_to_measure = urls
                st.session_state.results = {}
                st.session_state.idx = 0
                st.session_state.crawling = False
                st.success(f"Fetched {len(urls)} pages.")
            st.dataframe(pd.DataFrame({"URL": urls}), width='stretch')
    else:
        raw = st.text_area("Enter (or paste) list of URLs, one per line", value="", disabled=st.session_state.running or st.session_state.crawling)
        confirm_btn = st.button("Confirm list", disabled=st.session_state.running or st.session_state.crawling)
        if confirm_btn:
            urls = [u.strip() for u in raw.splitlines() if u.strip()]
            urls = [_norm_url(u) for u in urls]
            st.session_state.urls_to_measure = urls
            st.session_state.results = {}
            st.session_state.idx = 0
            st.success(f"Received {len(urls)} URLs.")
        st.dataframe(pd.DataFrame({"URL": urls_to_measure}), width='stretch')

with st.expander("Step 2: Batch PageSpeed audits", expanded=True):
    urls_to_measure = st.session_state.urls_to_measure
    colstart, colstop = st.columns([1,1])
    with colstart:
        start_audit = st.button("Start audits", disabled=st.session_state.running)
    with colstop:
        stop_audit = st.button("Stop", disabled=not st.session_state.running)
    if start_audit and not st.session_state.running:
        if not api_key:
            st.error("You must enter an API key.")
        elif not urls_to_measure:
            st.error("You must provide or crawl URLs before auditing.")
        else:
            st.session_state.running = True
            st.rerun()
    if stop_audit and st.session_state.running:
        st.session_state.running = False
    if st.session_state.running:
        results = {}
        max_workers = int(audit_workers)
        progress = st.progress(0.0, text="Auditing...")
        urls_total = len(urls_to_measure)
        completed = 0
        success_count = 0
        fail_count = 0
        using_backup = 0

        audit_devices = []
        if device_options[device_options.index(device)] == "Both (not recommended)":
            audit_devices = ["mobile", "desktop"]
        elif device_options[device_options.index(device)] == "Mobile (recommended for SEO)":
            audit_devices = ["mobile"]
        else:
            audit_devices = ["desktop"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_pagespeed_safe, u, dev, api_key, backup_api_key): (u, dev)
                for u in urls_to_measure for dev in audit_devices
            }
            for future in concurrent.futures.as_completed(futures):
                if not st.session_state.running:
                    break
                u, data, key_used = future.result()
                if "error" not in data:
                    results[u] = data
                    success_count += 1
                else:
                    fail_count += 1
                if key_used == "backup":
                    using_backup += 1
                completed += 1
                progress.progress(
                    completed / (urls_total * len(audit_devices)),
                    text=f"Completed {completed}/{urls_total * len(audit_devices)} audits"
                    + (f" | Using backup key for {using_backup}" if using_backup else "")
                )

        st.session_state.results = results
        st.session_state.running = False
        progress.empty()

        st.success(f"✅ Done! {success_count} success, {fail_count} failed."
                   + (f" ({using_backup} used backup key)" if using_backup else ""))

        now = time.strftime('%Y-%m-%d %H:%M:%S')
        audits_history = st.session_state.audits_history[-9:] if len(st.session_state.audits_history) >= 10 else st.session_state.audits_history
        record = {
            'timestamp': now,
            'device': device,
            'base_url': urls_to_measure[0] if urls_to_measure else '',
            'urls': [u for u in urls_to_measure if u in results],  # exclude fails
            'results': results
        }
        if not audits_history or audits_history[-1]['timestamp'] != now:
            audits_history.append(record)
        st.session_state.audits_history = audits_history[-10:]


urls_to_measure = st.session_state.urls_to_measure
results = st.session_state.results

if urls_to_measure and results:
    st.write("### Per-page / report results")
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    options = urls_to_measure or []
    st.selectbox("Select page", options=options, index=st.session_state.idx, key="picklist_select", disabled=st.session_state.running or st.session_state.crawling)
    st.session_state.idx = options.index(st.session_state.picklist_select) if st.session_state.picklist_select in options else 0

    cur_idx = st.session_state.idx
    cur_url = options[cur_idx] if options else ""
    cur = results.get(cur_url, {})
    st.write(f"**Page:** {cur_url}")
    if "error" in cur:
        st.error(f"Error auditing {cur_url}: {cur['error']}")
    else:
        overview = extract_overview_scores(cur)
        labels = list(overview.keys())
        scores = list(overview.values())
        st.markdown("#### Overall scores for 4 categories")
        cols = st.columns(4)
        for idx, col in enumerate(cols):
            s = scores[idx]
            color = score_color(s)
            label = labels[idx]
            display = f"<div style='text-align:center;padding:16px 0;border-radius:12px;background:{color};color:#222;font-weight:bold;font-size:1.5em;'>{s if s is not None else '—'}<br><span style='font-size:0.98em;'>{label}</span></div>"
            col.markdown(display, unsafe_allow_html=True)
        if any([s is None for s in scores]):
            st.warning("Some categories could not be measured. Check API key, access, or website configuration.")

        summary = extract_summary(cur)
        df_summary = pd.DataFrame([summary])
        def format_col(col):
            if col in ("Largest Contentful Paint (LCP)", "Speed Index", "Time to Interactive (TTI)", "First Contentful Paint (FCP)", "Total Blocking Time (TBT)"):
                return df_summary[col].apply(lambda v: f"{v/1000:.2f} s" if pd.notna(v) else "—")
            if col == "Cumulative Layout Shift (CLS)":
                return df_summary[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
            if col == "Performance Score":
                return df_summary[col].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "—")
            return df_summary[col]
        ordered_cols = [
            "Performance Score",
            "Largest Contentful Paint (LCP)",
            "Speed Index",
            "Time to Interactive (TTI)",
            "Cumulative Layout Shift (CLS)",
            "First Contentful Paint (FCP)",
            "Total Blocking Time (TBT)"
        ]
        df_disp = pd.DataFrame({c: format_col(c) for c in ordered_cols})
        st.write("#### Metrics overview")
        st.dataframe(df_disp, width='stretch')

        st.write("#### Details Issues (Opportunities + Diagnostics)")
        df_issues = extract_issues_table(cur)
        def fmt_value(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "—"
            s = str(v)
            try:
                num = float(str(v).replace(",", "").replace("ms", "").strip())
                if num <= 100000:
                    return f"{num:.0f} ms"
                else:
                    return f"{num/1024:.0f} KiB"
            except Exception:
                return s

        if not df_issues.empty:
            df_issues_disp = df_issues.copy()
            df_issues_disp["Value"] = df_issues_disp["Value"].apply(fmt_value)
            col_catgroup, col_issue, col_impact = st.columns([1,2,1])
            with col_catgroup:
                group_filter = st.multiselect("Filter Category Group", options=df_issues_disp["Category Group"].unique().tolist())
            with col_issue:
                issue_filter = st.multiselect("Filter Issue / Audit", options=df_issues_disp["Issue / Audit"].unique().tolist())
            with col_impact:
                impact_filter = st.multiselect("Filter Impact", options=df_issues_disp["Impact"].unique().tolist())

            filtered = df_issues_disp
            if group_filter:
                filtered = filtered[filtered["Category Group"].isin(group_filter)]
            if issue_filter:
                filtered = filtered[filtered["Issue / Audit"].isin(issue_filter)]
            if impact_filter:
                filtered = filtered[filtered["Impact"].isin(impact_filter)]
            st.dataframe(filtered, width='stretch')

        else:
            st.info("No Issues (Opportunities/Diagnostics) for this page.")

        try:
            filmstrip = _deep_get(cur, "lighthouseResult.audits.screenshot-thumbnails.details.items") or []
            if filmstrip:
                st.markdown("#### Page Loading Filmstrip (Screenshot Thumbnails)")
                imgs, captions = [], []
                for shot in filmstrip:
                    b64img = shot.get("data")
                    timing = shot.get("timing")
                    if b64img:
                        imgs.append(b64img)
                        captions.append(f"{timing} ms")
                if imgs:
                    cols = st.columns(len(imgs))
                    for i, (img, cap) in enumerate(zip(imgs, captions)):
                        with cols[i]:
                            st.image(img, caption=cap, width='stretch')
        except Exception as e:
            st.info(f"Could not retrieve loading screenshots: {e}")
        with st.expander("View raw JSON from API"):
            st.code(json.dumps(cur, indent=2, ensure_ascii=False))
    failed_urls = [u for u, d in results.items() if "error" in d]
    if failed_urls:
        st.markdown("### ❌ Failed URLs Summary")
        fail_data = []
        for u in failed_urls:
            err = results[u].get("error", "Unknown error")
            fail_data.append({"URL": u, "Error": err})
        df_fail = pd.DataFrame(fail_data)
        st.dataframe(df_fail, width='stretch')
