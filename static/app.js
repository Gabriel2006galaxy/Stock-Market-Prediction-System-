/* ── STATE ─────────────────────────────────────────────── */
let stocks = [];
let viewChart = null;
let cmpChart1 = null;
let cmpChart2 = null;
let predCharts = {};
let predActChart = null;

const COLORS = { RNN: "#ff6b35", LSTM: "#7b2fff", GRU: "#00d4aa" };

/* ── INIT ──────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
    loadStocks();
    initNav();
    document.getElementById("hamburger").addEventListener("click", () => {
        document.getElementById("sidebar").classList.toggle("open");
    });
});

/* ── NAVIGATION ────────────────────────────────────────── */
function initNav() {
    document.querySelectorAll(".nav-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const page = btn.dataset.page;
            switchPage(page);
            document.getElementById("sidebar").classList.remove("open");
        });
    });
}

function switchPage(name) {
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    document.getElementById("page-" + name).classList.add("active");
    document.querySelector(`[data-page="${name}"]`).classList.add("active");

    const titles = {
        dashboard: "Dashboard", add: "Add Stock", view: "View Stock",
        all: "All Stocks", news: "Portfolio News Feed", compare: "Compare Stocks",
        predict: "AI Prediction Engine", manage: "Manage Stocks"
    };
    document.getElementById("topbar-title").textContent = titles[name] || name;

    if (name === "all") renderAllStocks();
    if (name === "news") renderMarketNews();
    if (name === "predict" || name === "view" || name === "compare" || name === "manage") {
        populateSelects();
    }
}

/* ── STOCKS API ────────────────────────────────────────── */
async function loadStocks() {
    try {
        const res = await fetch("/api/stocks");
        stocks = await res.json();
        document.getElementById("stat-total-val").textContent = stocks.length;
    } catch (e) {
        console.error(e);
    }
}

function populateSelects() {
    const selects = [
        "view-select", "cmp-s1", "cmp-s2", "pred-select", "del-select"
    ];
    selects.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        const cur = el.value;
        el.innerHTML = '<option value="">— Select a stock —</option>';
        stocks.forEach(s => {
            const opt = document.createElement("option");
            opt.value = s.symbol;
            opt.textContent = `${s.name} (${s.symbol})`;
            el.appendChild(opt);
        });
        el.value = cur;
    });
}

/* ── ADD STOCK ─────────────────────────────────────────── */
async function addStock() {
    const name = document.getElementById("add-name").value.trim();
    const symbol = document.getElementById("add-symbol").value.trim().toUpperCase();
    const status = document.getElementById("add-status");

    if (!name || !symbol) {
        setStatus(status, "⚠️ Please fill in both company name and ticker symbol.", false);
        return;
    }

    try {
        const res = await fetch("/api/stocks", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, symbol })
        });
        const data = await res.json();
        console.log("addStock response", res.status, data);

        if (data.error) {
            setStatus(status, data.error, false);
            toast(data.error, false);
        } else if (data.success) {
            setStatus(status, `✅ ${data.name} (${data.symbol}) added successfully!`, true);
            document.getElementById("add-name").value = "";
            document.getElementById("add-symbol").value = "";
            await loadStocks();
            document.getElementById("stat-total-val").textContent = stocks.length;
            toast("Stock added to watchlist", true);
        }
    } catch (e) {
        console.error(e);
        setStatus(status, "❌ Network error. Please try again.", false);
    }
}

/* ── ALL STOCKS ────────────────────────────────────────── */
function renderAllStocks() {
    const grid = document.getElementById("all-list");
    grid.innerHTML = "";
    if (!stocks.length) {
        grid.innerHTML = '<p style="color:var(--muted);font-size:13px">No stocks in your watchlist yet.</p>';
        return;
    }
    stocks.forEach((s, i) => {
        const tile = document.createElement("div");
        tile.className = "stock-tile";
        tile.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:start">
        <div class="stock-tile-symbol">${s.symbol}</div>
        <div style="font-size:10px;color:var(--muted);background:var(--bg3);
                    padding:3px 7px;border-radius:3px;">#${i + 1}</div>
      </div>
      <div class="stock-tile-name">${s.name}</div>
    `;
        tile.addEventListener("click", () => {
            switchPage("view");
            setTimeout(() => {
                document.getElementById("view-select").value = s.symbol;
                loadView();
            }, 50);
        });
        grid.appendChild(tile);
    });
}

/* ── VIEW STOCK ────────────────────────────────────────── */
async function loadView() {
    const symbol = document.getElementById("view-select").value;
    if (!symbol) { 
        toast("⚠️ Please select a stock from the list", false); 
        return; 
    }

    show("view-loader"); 
    hide("view-result");

    try {
        const res = await fetch(`/api/view/${symbol}`);
        const data = await res.json();
        
        if (data.error) { 
            toast(`❌ ${data.error}`, false); 
            hide("view-loader"); 
            return; 
        }

        if (!data.candles || data.candles.length === 0) {
            toast(`❌ No chart data available for ${symbol}`, false);
            hide("view-loader");
            return;
        }

        hide("view-loader"); 
        show("view-result");
        fetchNews(symbol, "view-news");

        const stock = stocks.find(s => s.symbol === symbol) || { name: symbol };
        const sign = data.change >= 0 ? "+" : "";
        const cls = data.trend === "up" ? "up" : "down";
        const arrow = data.trend === "up" ? "▲" : "▼";

        document.getElementById("view-chart-header").innerHTML = `
      <span class="chart-symbol">${symbol}</span>
      <span style="font-size:14px;color:var(--muted)">${stock.name}</span>
      <span class="chart-change ${cls}">${arrow} ${sign}${data.change} (${sign}${data.percent}%)</span>
    `;

        // destroy old chart
        if (viewChart) { viewChart.remove(); viewChart = null; }
        const container = document.getElementById("view-chart");
        container.innerHTML = "";

        viewChart = LightweightCharts.createChart(container, chartOptions("#view-chart"));
        const candleSeries = viewChart.addCandlestickSeries({
            upColor: "#00ff88", downColor: "#ff4455",
            borderUpColor: "#00ff88", borderDownColor: "#ff4455",
            wickUpColor: "#00ff88", wickDownColor: "#ff4455"
        });
        candleSeries.setData(data.candles.map(c => ({
            time: c.date, open: c.open, high: c.high, low: c.low, close: c.close
        })));
        viewChart.timeScale().fitContent();

        // table
        const tbody = document.querySelector("#view-table tbody");
        tbody.innerHTML = "";
        data.candles.forEach(c => {
            const chg = c.close - c.open;
            const cls = chg >= 0 ? "up-val" : "down-val";
            tbody.insertAdjacentHTML("beforeend", `
        <tr>
          <td>${c.date}</td>
          <td>${c.open.toFixed(2)}</td>
          <td>${c.high.toFixed(2)}</td>
          <td>${c.low.toFixed(2)}</td>
          <td class="${cls}">${c.close.toFixed(2)}</td>
          <td>${c.volume.toLocaleString()}</td>
        </tr>`);
        });

    } catch (e) {
        console.error(e);
        hide("view-loader");
        toast("❌ Failed to load chart data. Check your connection.", false);
    }
}

/* ── COMPARE ───────────────────────────────────────────── */
async function compareStocks() {
    const s1 = document.getElementById("cmp-s1").value;
    const s2 = document.getElementById("cmp-s2").value;
    
    if (!s1 || !s2) { 
        toast("⚠️ Please select both stocks to compare", false); 
        return; 
    }
    if (s1 === s2) { 
        toast("❌ Please select two different stocks", false); 
        return; 
    }

    show("cmp-loader"); 
    hide("cmp-result");

    try {
        const res = await fetch(`/api/compare?s1=${s1}&s2=${s2}`);
        const data = await res.json();
        
        if (data.error) { 
            toast(`❌ ${data.error}`, false); 
            hide("cmp-loader"); 
            return; 
        }
        if (!data.stock1 || !data.stock2) {
            toast(`❌ Could not load data for one or both stocks`, false);
            hide("cmp-loader");
            return;
        }

        hide("cmp-loader"); 
        show("cmp-result");
        fetchNews(s1, "cmp-news-1");
        fetchNews(s2, "cmp-news-2");

        renderComparePanel("cmp-chart-1", "cmp-label-1", "cmp-stats-1", s1, data.stock1, "#00d4aa", cmpChart1, c => cmpChart1 = c);
        renderComparePanel("cmp-chart-2", "cmp-label-2", "cmp-stats-2", s2, data.stock2, "#ff6b35", cmpChart2, c => cmpChart2 = c);

    } catch (e) {
        console.error(e);
        hide("cmp-loader");
        toast("❌ Failed to load comparison data. Check connection.", false);
    }
}

function renderComparePanel(chartId, labelId, statsId, symbol, data, color, oldChart, setChart) {
    const stock = stocks.find(s => s.symbol === symbol) || { name: symbol };
    const sign = data.change >= 0 ? "+" : "";
    const cls = data.trend === "up" ? "up" : "down";
    const arrow = data.trend === "up" ? "▲" : "▼";

    document.getElementById(labelId).innerHTML = `
    <span style="color:${color}">${symbol}</span>
    <span style="font-size:12px;color:var(--muted);margin-left:10px">${stock.name}</span>
  `;

    if (oldChart) { oldChart.remove(); }
    const container = document.getElementById(chartId);
    container.innerHTML = "";

    const chart = LightweightCharts.createChart(container, chartOptions(chartId));
    const cs = chart.addCandlestickSeries({
        upColor: color, downColor: "#ff4455",
        borderUpColor: color, borderDownColor: "#ff4455",
        wickUpColor: color, wickDownColor: "#ff4455"
    });
    cs.setData(data.candles.map(c => ({
        time: c.date, open: c.open, high: c.high, low: c.low, close: c.close
    })));
    chart.timeScale().fitContent();
    setChart(chart);

    document.getElementById(statsId).innerHTML = `
    <div class="cmp-stat">
      <div class="cmp-stat-label">CHANGE</div>
      <div class="cmp-stat-val ${cls}">${arrow} ${sign}${data.change} (${sign}${data.percent}%)</div>
    </div>
    <div class="cmp-stat">
      <div class="cmp-stat-label">PERIOD HIGH</div>
      <div class="cmp-stat-val">${data.high}</div>
    </div>
    <div class="cmp-stat">
      <div class="cmp-stat-label">PERIOD LOW</div>
      <div class="cmp-stat-val">${data.low}</div>
    </div>
    <div class="cmp-stat">
      <div class="cmp-stat-label">AVG VOLUME</div>
      <div class="cmp-stat-val">${data.avg_vol.toLocaleString()}</div>
    </div>
  `;
}

/* ── PREDICT ───────────────────────────────────────────── */
async function runPrediction() {
    const symbol = document.getElementById("pred-select").value;
    if (!symbol) { 
        toast("⚠️ Please select a stock to predict", false); 
        return; 
    }

    const selected = [];
    if (document.getElementById("use-rnn").checked) selected.push("RNN");
    if (document.getElementById("use-lstm").checked) selected.push("LSTM");
    if (document.getElementById("use-gru").checked) selected.push("GRU");
    if (!selected.length) { 
        toast("⚠️ Select at least one AI model (RNN, LSTM, or GRU)", false); 
        return; 
    }

    show("pred-loader"); 
    hide("pred-result");
    document.getElementById("pred-btn").disabled = true;

    // progress animation
    const fill = document.getElementById("pred-progress");
    const status = document.getElementById("pred-status");
    let pct = 0;
    const messages = [
        "Fetching 4 years of market data…",
        "Running Genetic Algorithm optimisation…",
        "Training deep neural networks…",
        "Evaluating accuracy metrics…",
        "Forecasting next 15 trading days…"
    ];
    let msgIdx = 0;
    const progInterval = setInterval(() => {
        pct = Math.min(pct + Math.random() * 4, 92);
        fill.style.width = pct + "%";
        if (pct > msgIdx * 20 && msgIdx < messages.length) {
            status.textContent = messages[msgIdx++];
        }
    }, 600);

    try {
        const res = await fetch(`/api/predict/${symbol}?models=${selected.join(",")}`);
        const data = await res.json();

        clearInterval(progInterval);
        fill.style.width = "100%";
        hide("pred-loader");
        document.getElementById("pred-btn").disabled = false;

        if (data.error) { 
            toast(`❌ Prediction failed: ${data.error}`, false); 
            return; 
        }

        show("pred-result");
        window.lastPredictionData = data;
        window.lastPredictionSelected = selected;
        renderPrediction(data, selected, symbol);
        fetchNews(symbol, "pred-news");

    } catch (e) {
        console.error(e);
        clearInterval(progInterval);
        hide("pred-loader");
        document.getElementById("pred-btn").disabled = false;
        toast("❌ Prediction server error. Please try again.", false);
    }
}

function renderPrediction(data, selected, symbol) {
    const stock = stocks.find(s => s.symbol === symbol) || { name: symbol };

    // ── CONSENSUS ──
    const cMap = {
        ALL_UP: { text: "🧠 All models agree — UPWARD TREND 📈", cls: "up" },
        ALL_DOWN: { text: "🧠 All models agree — DOWNWARD TREND 📉", cls: "down" },
        MAJORITY_UP: { text: "🧠 Majority predict UPWARD TREND 📈", cls: "up" },
        MAJORITY_DOWN: { text: "🧠 Majority predict DOWNWARD TREND 📉", cls: "down" },
        SPLIT: { text: "🧠 Models are split — No clear trend", cls: "split" },
    };
    const c = cMap[data.consensus] || { text: "—", cls: "split" };
    const banner = document.getElementById("consensus-banner");
    banner.className = "consensus-banner " + c.cls;
    banner.textContent = c.text;

    // ── LEADERBOARD ──
    const lb = document.getElementById("leaderboard");
    lb.innerHTML = `
    <div class="lb-row header">
      <span>ALGO</span><span>MAE</span><span>RMSE</span><span>DIR ACC</span><span></span>
    </div>
  `;
    selected.forEach(tag => {
        const m = data.accuracy[tag];
        const cls = tag === data.best ? " best" : "";
        lb.insertAdjacentHTML("beforeend", `
      <div class="lb-row${cls}" style="border-left:3px solid ${COLORS[tag]}">
        <span class="lb-tag" style="color:${COLORS[tag]}">${tag}</span>
        <span>${m.mae}</span>
        <span>${m.rmse}</span>
        <span>${m.da}%</span>
        <span class="lb-badge">${tag === data.best ? "🏆 BEST" : ""}</span>
      </div>`);
    });

    // ── CHARTS ──
    // destroy old
    Object.values(predCharts).forEach(c => c && c.remove());
    predCharts = {};
    if (predActChart) { predActChart.remove(); predActChart = null; }

    const chartsWrap = document.getElementById("pred-charts");
    chartsWrap.innerHTML = "";

    // actual panel
    const actPanel = document.createElement("div");
    actPanel.className = "pred-chart-panel";
    actPanel.innerHTML = `
    <div class="pred-chart-title">
      <span>📊 Actual — ${stock.name} (Last 20 Days)</span>
    </div>
    <div id="pact-chart" class="pred-chart-box"></div>
  `;
    chartsWrap.appendChild(actPanel);

    setTimeout(() => {
        predActChart = LightweightCharts.createChart(
            document.getElementById("pact-chart"),
            chartOptions("pact-chart", 260)
        );
        const cs = predActChart.addCandlestickSeries({
            upColor: "#00ff88", downColor: "#ff4455",
            borderUpColor: "#00ff88", borderDownColor: "#ff4455",
            wickUpColor: "#00ff88", wickDownColor: "#ff4455"
        });
        cs.setData(data.actual.map(c => ({
            time: c.date, open: c.open, high: c.high, low: c.low, close: c.close
        })));
        predActChart.timeScale().fitContent();
    }, 50);

    // predicted panels
    selected.forEach((tag, idx) => {
        const rows = data.predicted[tag];
        const acc = data.accuracy[tag];
        const color = COLORS[tag];
        const chg = rows[rows.length - 1].close - rows[0].close;
        const arrow = chg >= 0 ? "▲" : "▼";
        const cls = chg >= 0 ? "var(--green)" : "var(--red)";

        const panel = document.createElement("div");
        panel.className = "pred-chart-panel";
        panel.style.borderTopColor = color;
        panel.innerHTML = `
      <div class="pred-chart-title" style="color:${color}">
        <span>${tag} — Next 15 Days</span>
        <span style="font-size:10px;color:var(--muted)">
          RMSE:${acc.rmse} MAE:${acc.mae} DA:${acc.da}%
        </span>
      </div>
      <div id="pc-${tag}" class="pred-chart-box"></div>
      <div style="margin-top:8px;font-size:12px;color:${cls}">
        ${arrow} ${chg >= 0 ? "+" : ""}${chg.toFixed(2)} forecast change
      </div>
    `;
        chartsWrap.appendChild(panel);

        setTimeout(() => {
            const chart = LightweightCharts.createChart(
                document.getElementById(`pc-${tag}`),
                chartOptions(`pc-${tag}`, 260)
            );
            const cs = chart.addCandlestickSeries({
                upColor: color, downColor: "#ff4455",
                borderUpColor: color, borderDownColor: "#ff4455",
                wickUpColor: color, wickDownColor: "#ff4455"
            });
            cs.setData(rows.map(r => ({
                time: r.date || r.Date,
                open: r.open || r.Open,
                high: r.high || r.High,
                low: r.low || r.Low,
                close: r.close || r.Close
            })));
            chart.timeScale().fitContent();
            predCharts[tag] = chart;
        }, 60 + idx * 40);
    });

    // ── TABLES ──
    const tablesWrap = document.getElementById("pred-tables");
    tablesWrap.innerHTML = "";

    selected.forEach(tag => {
        const rows = data.predicted[tag];
        const color = COLORS[tag];
        const panel = document.createElement("div");
        panel.className = "pred-table-panel";
        panel.style.borderTop = `3px solid ${color}`;
        panel.innerHTML = `
      <div class="pred-table-title" style="color:${color}">${tag} — Predicted Prices</div>
      <div class="data-table-wrap">
        <table class="data-table">
          <thead><tr>
            <th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th>
          </tr></thead>
          <tbody>
            ${rows.map(r => {
            const chg = r.close - r.open;
            const cls = chg >= 0 ? "up-val" : "down-val";
            return `<tr>
                <td>${r.date || r.Date}</td>
                <td>${(r.open || r.Open).toFixed(2)}</td>
                <td>${(r.high || r.High).toFixed(2)}</td>
                <td>${(r.low || r.Low).toFixed(2)}</td>
                <td class="${cls}">${(r.close || r.Close).toFixed(2)}</td>
                <td>${(r.volume || r.Volume || 0).toLocaleString()}</td>
              </tr>`;
        }).join("")}
          </tbody>
        </table>
      </div>
    `;
        tablesWrap.appendChild(panel);
    });
}

/* ── DELETE ────────────────────────────────────────────── */
async function deleteStock() {
    const symbol = document.getElementById("del-select").value;
    const status = document.getElementById("del-status");
    if (!symbol) { 
        setStatus(status, "⚠️ Select a stock to delete first.", false); 
        return; 
    }

    if (!confirm(`⚠️ Are you sure you want to delete ${symbol} from your watchlist?`)) return;

    try {
        const res = await fetch(`/api/stocks/${symbol}`, { method: "DELETE" });
        const data = await res.json();
        
        if (res.ok && data.success) {
            toast(`✅ ${symbol} deleted from watchlist`, true);
            setStatus(status, `✅ ${symbol} deleted successfully.`, true);
            await loadStocks();
            populateSelects();
            document.getElementById("del-select").value = "";
        } else {
            setStatus(status, `❌ Failed to delete ${symbol}`, false);
            toast(`❌ Failed to delete ${symbol}`, false);
        }
    } catch (e) {
        console.error(e);
        setStatus(status, "❌ Network error. Failed to delete.", false);
    }
}


async function confirmDeleteAll() {
    const status = document.getElementById("del-all-status");
    if (!confirm("⚠️ Delete ALL stocks permanently? This cannot be undone.")) return;
    if (!confirm("🔴 Are you absolutely sure? All watchlist items will be removed.")) return;

    try {
        const res = await fetch("/api/stocks/all", { method: "DELETE" });
        const data = await res.json();
        
        if (res.ok && data.success) {
            toast("✅ All stocks deleted from watchlist", true);
            setStatus(status, "✅ All stocks deleted permanently.", true);
            await loadStocks();
            populateSelects();
            document.getElementById("stat-total-val").textContent = "0";
        } else {
            setStatus(status, "❌ Failed to delete all stocks", false);
            toast("❌ Failed to delete all stocks", false);
        }
    } catch (e) {
        console.error(e);
        setStatus(status, "❌ Network error. Delete failed.", false);
    }
}


/* ── CHART OPTIONS ─────────────────────────────────────── */
function chartOptions(containerId, height) {
    return {
        autoSize: true,
        layout: { background: { color: "#0e0e1c" }, textColor: "#6a6a8a" },
        grid: {
            vertLines: { color: "rgba(255,255,255,0.04)" },
            horzLines: { color: "rgba(255,255,255,0.04)" }
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: "rgba(255,255,255,0.07)" },
        timeScale: {
            borderColor: "rgba(255,255,255,0.07)",
            timeVisible: true
        }
    };
}

/* ── UTILS ─────────────────────────────────────────────── */
function show(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove("hidden");
}
function hide(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add("hidden");
}
function setStatus(el, msg, ok) {
    el.textContent = msg;
    el.className = "form-status " + (ok ? "ok" : "err");
}
function toast(msg, ok) {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.className = "toast show " + (ok ? "ok" : "err");
    setTimeout(() => t.classList.remove("show"), 3000);
}

/* ── EXPORT CSV ────────────────────────────────────────── */
function exportPredictionsCSV() {
    const data = window.lastPredictionData;
    const selected = window.lastPredictionSelected;
    if (!data || !selected) { toast("No predictions to export", false); return; }

    let csvContent = "data:text/csv;charset=utf-8,Algo,Date,Open,High,Low,Close,Volume\n";
    selected.forEach(tag => {
        data.predicted[tag].forEach(r => {
            const date = r.date || r.Date;
            const o = (r.open || r.Open).toFixed(2);
            const h = (r.high || r.High).toFixed(2);
            const l = (r.low || r.Low).toFixed(2);
            const c = (r.close || r.Close).toFixed(2);
            const v = (r.volume || r.Volume || 0);
            csvContent += `${tag},${date},${o},${h},${l},${c},${v}\n`;
        });
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "predictions.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/* ── NEWS MODULE ───────────────────────────────────────── */
async function fetchNews(symbol, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = `<div class="spinner"></div><span style="margin-left:8px;font-size:12px;color:var(--muted)">Loading news...</span>`;
    try {
        const res = await fetch(`/api/news/${symbol}`);
        const news = await res.json();
        if (news.error || news.length === 0) {
            container.innerHTML = `<p style="font-size:12px;color:var(--muted)">No recent news found.</p>`;
            return;
        }
        let html = '<div style="display:flex;flex-direction:column;gap:12px;margin-top:14px;">';
        news.forEach(n => {
            let d = n.pubDate ? new Date(n.pubDate).toLocaleDateString() : 'Recent';
            html += `
                <div style="background:var(--bg3);padding:14px;border-radius:8px;border:1px solid var(--border);transition:all 0.2s;">
                    <a href="${n.link}" target="_blank" style="color:var(--accent);text-decoration:none;font-size:13px;font-weight:bold;margin-bottom:6px;display:block;">${n.title}</a>
                    <p style="color:var(--text);font-size:11px;line-height:1.5;">${n.summary}</p>
                    <div style="color:var(--muted);font-size:10px;margin-top:8px;">${d}</div>
                </div>
            `;
        });
        html += '</div>';
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = `<p style="font-size:12px;color:var(--red)">Failed to load news.</p>`;
    }
}

async function renderMarketNews() {
    const container = document.getElementById("market-news-feed");
    container.innerHTML = `<div class="loader"><div class="spinner"></div><span style="margin-left:8px;">Aggregating full portfolio news...</span></div>`;

    if (!stocks || stocks.length === 0) {
        container.innerHTML = `<p style="color:var(--muted)">Add some stocks to see news.</p>`;
        return;
    }

    let allNews = [];
    await Promise.all(stocks.map(async s => {
        try {
            const res = await fetch(`/api/news/${s.symbol}`);
            const news = await res.json();
            if (!news.error) {
                news.forEach(n => { n.stockSymbol = s.symbol; n.stockName = s.name; });
                allNews = allNews.concat(news);
            }
        } catch (e) { }
    }));

    allNews.sort((a, b) => {
        if (!a.pubDate || !b.pubDate) return 0;
        return new Date(b.pubDate) - new Date(a.pubDate);
    });

    if (allNews.length === 0) {
        container.innerHTML = `<p style="color:var(--muted)">No news found across your portfolio.</p>`;
        return;
    }

    let html = '<div style="display:flex;flex-direction:column;gap:16px;max-width:800px;">';
    allNews.forEach(n => {
        let d = n.pubDate ? new Date(n.pubDate).toLocaleString() : 'Recent';
        html += `
            <div style="background:var(--bg2);padding:18px;border-radius:10px;border:1px solid var(--border);transition:all 0.2s;">
                <div style="margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;">
                    <span style="background:rgba(0,229,255,0.1);color:var(--accent);padding:6px 12px;border-radius:5px;font-size:12px;font-weight:bold;">${n.stockName} (${n.stockSymbol})</span>
                    <span style="color:var(--muted);font-size:12px;">${d}</span>
                </div>
                <a href="${n.link}" target="_blank" style="color:#fff;text-decoration:none;font-size:17px;font-family:'Syne',sans-serif;font-weight:bold;margin-bottom:8px;display:block;">${n.title}</a>
                <p style="color:var(--text);font-size:13px;line-height:1.6;">${n.summary}</p>
            </div>
        `;
    });
    html += '</div>';
    container.innerHTML = html;
}