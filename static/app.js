/* ── STATE ─────────────────────────────────────────────── */
let stocks = [];
let viewChart = null;
let cmpChart1 = null;
let cmpChart2 = null;
let predCharts = {};
let predActChart = null;

// Tracks the currently selected symbol for each searchable selector
let selectedSymbols = {
    view: "",
    cmpS1: "",
    cmpS2: "",
    pred: "",
    del: ""
};

const COLORS = { RNN: "#ff6b35", LSTM: "#7b2fff", GRU: "#00d4aa" };

/* ── INIT ──────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
    loadStocks();
    initNav();
    document.getElementById("hamburger").addEventListener("click", () => {
        document.getElementById("sidebar").classList.toggle("open");
    });
    // Close any open dropdowns when clicking outside
    document.addEventListener("click", (e) => {
        if (!e.target.closest(".search-select-wrap")) {
            document.querySelectorAll(".ss-dropdown").forEach(d => d.classList.add("hidden"));
        }
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
    } catch (e) { console.error(e); }
}

/* ── SEARCHABLE SELECT ─────────────────────────────────────
   Each selector has:
     - a text input where the user types to filter
     - a dropdown div that shows matching stocks
     - a hidden selected value tracked in selectedSymbols{}
   
   Usage: buildSearchSelect("my-wrap-id", "view")
   Then read: selectedSymbols["view"]
────────────────────────────────────────────────────────── */
function buildSearchSelect(wrapId, key, placeholder) {
    const wrap = document.getElementById(wrapId);
    if (!wrap) return;
    wrap.classList.add("search-select-wrap");
    wrap.innerHTML = `
        <input 
            type="text" 
            class="ss-input styled-select" 
            id="ss-input-${key}" 
            placeholder="${placeholder || '— Type to search stock —'}"
            autocomplete="off"
        />
        <div class="ss-dropdown hidden" id="ss-drop-${key}"></div>
        <div class="ss-selected hidden" id="ss-selected-${key}"></div>
    `;
    const input = document.getElementById(`ss-input-${key}`);
    const drop  = document.getElementById(`ss-drop-${key}`);

    input.addEventListener("input", () => {
        const q = input.value.trim().toLowerCase();
        renderDropdown(key, q);
        drop.classList.remove("hidden");
    });

    input.addEventListener("focus", () => {
        const q = input.value.trim().toLowerCase();
        renderDropdown(key, q);
        drop.classList.remove("hidden");
    });
}

function renderDropdown(key, query) {
    const drop = document.getElementById(`ss-drop-${key}`);
    if (!drop) return;

    const filtered = query
        ? stocks.filter(s =>
            s.symbol.toLowerCase().includes(query) ||
            s.name.toLowerCase().includes(query))
        : stocks;

    if (!filtered.length) {
        drop.innerHTML = `<div class="ss-empty">No stocks match "${query}"</div>`;
        return;
    }

    drop.innerHTML = filtered.map(s => {
        const exchange = s.symbol.endsWith(".NS") ? "NSE"
                       : s.symbol.endsWith(".BO") ? "BSE" : "US";
        const badgeColor = exchange === "NSE" ? "var(--accent)"
                         : exchange === "BSE" ? "var(--accent3)" : "var(--accent2)";
        return `
            <div class="ss-item" onclick="selectStock('${key}', '${s.symbol}', '${s.name.replace(/'/g, "\\'")}')">
                <span class="ss-sym">${s.symbol}</span>
                <span class="ss-name">${s.name}</span>
                <span class="ss-badge" style="color:${badgeColor};border-color:${badgeColor}">${exchange}</span>
            </div>
        `;
    }).join("");
}

function selectStock(key, symbol, name) {
    selectedSymbols[key] = symbol;
    const input = document.getElementById(`ss-input-${key}`);
    const drop  = document.getElementById(`ss-drop-${key}`);
    if (input) input.value = `${symbol} — ${name}`;
    if (drop)  drop.classList.add("hidden");
}

function clearSearchSelect(key) {
    selectedSymbols[key] = "";
    const input = document.getElementById(`ss-input-${key}`);
    if (input) input.value = "";
}

function populateSelects() {
    // Build each searchable select if not already built
    // view page
    if (!document.getElementById("ss-input-view")) {
        buildSearchSelect("view-select-wrap", "view", "— Type to search stock —");
    }
    // compare page
    if (!document.getElementById("ss-input-cmpS1")) {
        buildSearchSelect("cmp-s1-wrap", "cmpS1", "— First stock —");
    }
    if (!document.getElementById("ss-input-cmpS2")) {
        buildSearchSelect("cmp-s2-wrap", "cmpS2", "— Second stock —");
    }
    // predict page
    if (!document.getElementById("ss-input-pred")) {
        buildSearchSelect("pred-select-wrap", "pred", "— Type to search stock —");
    }
    // manage page
    if (!document.getElementById("ss-input-del")) {
        buildSearchSelect("del-select-wrap", "del", "— Select a stock to delete —");
    }
}

/* ── ADD STOCK ─────────────────────────────────────────── */
async function addStock() {
    const name      = document.getElementById("add-name").value.trim();
    const rawSymbol = document.getElementById("add-symbol").value.trim();
    const status    = document.getElementById("add-status");

    if (!name) {
        setStatus(status, "⚠ Please enter the company name.", false); return;
    }
    if (!rawSymbol) {
        setStatus(status, "⚠ Please enter a ticker symbol.", false); return;
    }

    const symbol = rawSymbol.toUpperCase();

    if (/\s/.test(symbol)) {
        setStatus(status, "❌ Ticker cannot contain spaces. Example: INFY.NS", false); return;
    }
    if (!/^[A-Z0-9.\-&^]+$/.test(symbol)) {
        setStatus(status, "❌ Ticker has invalid characters. Use letters, numbers, dots or hyphens only.", false); return;
    }

   if (rawSymbol !== rawSymbol.toUpperCase()) {
    setStatus(status, "⚠ Ticker auto-corrected to uppercase: " + symbol, false);
    document.getElementById("add-symbol").value = symbol;
    
}

    const KNOWN_US = ["AAPL","MSFT","GOOG","GOOGL","AMZN","META","TSLA","NVDA","NFLX","AMD","INTC","CRM","ORCL","IBM","UBER","LYFT","SNAP","SPOT"];
    const looksIndian = /^[A-Z]{2,15}$/.test(symbol) && !KNOWN_US.includes(symbol);
    if (looksIndian && !symbol.endsWith(".NS") && !symbol.endsWith(".BO")) {
        setStatus(status, `⚠ Is this an Indian stock? Try ${symbol}.NS (NSE) or ${symbol}.BO (BSE) instead.`, false);
        return;
    }

    try {
        const res = await fetch("/api/stocks", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, symbol })
        });
        const data = await res.json();
        if (data.error) {
            let msg = data.error;
            if (data.suggestion) {
                msg += ` — Did you mean: ${data.suggestion}?`;
            }
            if (msg.toLowerCase().includes("duplicate") || msg.toLowerCase().includes("already")) {
                setStatus(status, `❌ ${symbol} is already in your watchlist.`, false);
            } else {
                setStatus(status, "❌ " + msg, false);
            }
        } else {
            setStatus(status, `✅ ${data.name} (${data.symbol}) added!`, true);
            document.getElementById("add-name").value = "";
            document.getElementById("add-symbol").value = "";
            await loadStocks();
            document.getElementById("stat-total-val").textContent = stocks.length;
            toast("Stock added successfully", true);
        }
    } catch (e) {
        setStatus(status, "❌ Server error. Please try again.", false);
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
                <div style="font-size:10px;color:var(--muted);background:var(--bg3);padding:3px 7px;border-radius:3px;">#${i + 1}</div>
            </div>
            <div class="stock-tile-name">${s.name}</div>`;
        tile.addEventListener("click", () => {
            switchPage("view");
            setTimeout(() => {
                selectStock("view", s.symbol, s.name);
                loadView();
            }, 50);
        });
        grid.appendChild(tile);
    });
}

/* ── VIEW STOCK ────────────────────────────────────────── */
async function loadView() {
    const symbol = selectedSymbols.view;
    if (!symbol) { toast("⚠ Please select a stock first", false); return; }
    show("view-loader"); hide("view-result");
    try {
        const res  = await fetch(`/api/view/${symbol}`);
        const data = await res.json();
        if (data.error) {
            toast("❌ " + data.error, false);
            hide("view-loader");
            return;
        }
        hide("view-loader"); show("view-result");
        fetchNews(symbol, "view-news");
        const stock = stocks.find(s => s.symbol === symbol) || { name: symbol };
        const sign  = data.change >= 0 ? "+" : "";
        const cls   = data.trend === "up" ? "up" : "down";
        const arrow = data.trend === "up" ? "▲" : "▼";
        document.getElementById("view-chart-header").innerHTML = `
            <span class="chart-symbol">${symbol}</span>
            <span style="font-size:14px;color:var(--muted)">${stock.name}</span>
            <span class="chart-change ${cls}">${arrow} ${sign}${data.change} (${sign}${data.percent}%)</span>`;
        if (viewChart) { viewChart.remove(); viewChart = null; }
        const container = document.getElementById("view-chart");
        container.innerHTML = "";
        viewChart = LightweightCharts.createChart(container, chartOptions("#view-chart"));
        const candleSeries = viewChart.addCandlestickSeries({
            upColor: "#00ff88", downColor: "#ff4455",
            borderUpColor: "#00ff88", borderDownColor: "#ff4455",
            wickUpColor: "#00ff88", wickDownColor: "#ff4455"
        });
        candleSeries.setData(data.candles.map(c => ({ time: c.date, open: c.open, high: c.high, low: c.low, close: c.close })));
        viewChart.timeScale().fitContent();
        const tbody = document.querySelector("#view-table tbody");
        tbody.innerHTML = "";
        data.candles.forEach(c => {
            const chg = c.close - c.open;
            tbody.insertAdjacentHTML("beforeend", `<tr>
                <td>${c.date}</td><td>${c.open.toFixed(2)}</td><td>${c.high.toFixed(2)}</td>
                <td>${c.low.toFixed(2)}</td><td class="${chg>=0?"up-val":"down-val"}">${c.close.toFixed(2)}</td>
                <td>${c.volume.toLocaleString()}</td></tr>`);
        });
    } catch (e) {
        hide("view-loader");
        toast("❌ Failed to load stock data. Please try again.", false);
    }
}

/* ── COMPARE ───────────────────────────────────────────── */
async function compareStocks() {
    const s1 = selectedSymbols.cmpS1;
    const s2 = selectedSymbols.cmpS2;
    if (!s1 && !s2) { toast("⚠ Please select both stocks to compare", false); return; }
    if (!s1)        { toast("⚠ Please select the first stock", false); return; }
    if (!s2)        { toast("⚠ Please select the second stock", false); return; }
    if (s1 === s2)  { toast("⚠ Please select two different stocks", false); return; }
    show("cmp-loader"); hide("cmp-result");
    try {
        const res  = await fetch(`/api/compare?s1=${s1}&s2=${s2}`);
        const data = await res.json();
        if (data.error) { toast("❌ " + data.error, false); hide("cmp-loader"); return; }
        hide("cmp-loader"); show("cmp-result");
        fetchNews(s1, "cmp-news-1"); fetchNews(s2, "cmp-news-2");
        renderComparePanel("cmp-chart-1","cmp-label-1","cmp-stats-1",s1,data.stock1,"#00d4aa",cmpChart1,c=>cmpChart1=c);
        renderComparePanel("cmp-chart-2","cmp-label-2","cmp-stats-2",s2,data.stock2,"#ff6b35",cmpChart2,c=>cmpChart2=c);
    } catch (e) {
        hide("cmp-loader");
        toast("❌ Failed to load comparison. Please try again.", false);
    }
}

function renderComparePanel(chartId, labelId, statsId, symbol, data, color, oldChart, setChart) {
    const stock = stocks.find(s => s.symbol === symbol) || { name: symbol };
    const sign  = data.change >= 0 ? "+" : "";
    const cls   = data.trend === "up" ? "up" : "down";
    const arrow = data.trend === "up" ? "▲" : "▼";
    document.getElementById(labelId).innerHTML = `
        <span style="color:${color}">${symbol}</span>
        <span style="font-size:12px;color:var(--muted);margin-left:10px">${stock.name}</span>`;
    if (oldChart) oldChart.remove();
    const container = document.getElementById(chartId);
    container.innerHTML = "";
    const chart = LightweightCharts.createChart(container, chartOptions(chartId));
    const cs = chart.addCandlestickSeries({
        upColor: color, downColor: "#ff4455",
        borderUpColor: color, borderDownColor: "#ff4455",
        wickUpColor: color, wickDownColor: "#ff4455"
    });
    cs.setData(data.candles.map(c => ({ time: c.date, open: c.open, high: c.high, low: c.low, close: c.close })));
    chart.timeScale().fitContent();
    setChart(chart);
    document.getElementById(statsId).innerHTML = `
        <div class="cmp-stat"><div class="cmp-stat-label">CHANGE</div>
            <div class="cmp-stat-val ${cls}">${arrow} ${sign}${data.change} (${sign}${data.percent}%)</div></div>
        <div class="cmp-stat"><div class="cmp-stat-label">PERIOD HIGH</div>
            <div class="cmp-stat-val">${data.high}</div></div>
        <div class="cmp-stat"><div class="cmp-stat-label">PERIOD LOW</div>
            <div class="cmp-stat-val">${data.low}</div></div>
        <div class="cmp-stat"><div class="cmp-stat-label">AVG VOLUME</div>
            <div class="cmp-stat-val">${data.avg_vol.toLocaleString()}</div></div>`;
}

/* ── PREDICT ───────────────────────────────────────────── */
async function runPrediction() {
    const symbol = selectedSymbols.pred;
    if (!symbol) { toast("⚠ Please select a stock first", false); return; }

    const selected = [];
    if (document.getElementById("use-rnn").checked)  selected.push("RNN");
    if (document.getElementById("use-lstm").checked) selected.push("LSTM");
    if (document.getElementById("use-gru").checked)  selected.push("GRU");
    if (!selected.length) { toast("⚠ Select at least one algorithm", false); return; }

    show("pred-loader"); hide("pred-result");
    document.getElementById("pred-btn").disabled = true;

    const fill   = document.getElementById("pred-progress");
    const status = document.getElementById("pred-status");
    fill.style.width = "0%";

    const messages = [
        "Fetching 4 years of market data…",
        "Running Genetic Algorithm optimisation…",
        "Training deep neural networks…",
        "Evaluating accuracy metrics…",
        "Forecasting next 15 trading days…"
    ];
    let pct = 0, msgIdx = 0;
    const progInterval = setInterval(() => {
        pct = Math.min(pct + Math.random() * 2.5, 90);
        fill.style.width = pct + "%";
        if (pct > msgIdx * 20 && msgIdx < messages.length) {
            status.textContent = messages[msgIdx++];
        }
    }, 800);

    try {
        const res  = await fetch(`/api/predict/${symbol}?models=${selected.join(",")}`);
        const data = await res.json();

        clearInterval(progInterval);
        fill.style.width = "100%";
        hide("pred-loader");
        document.getElementById("pred-btn").disabled = false;

        if (data.error) {
            toast("❌ " + data.error, false);
            status.textContent = data.error;
            return;
        }

        if (data.skipped && data.skipped.length) {
            toast(`⚠ ${data.skipped.join(", ")} skipped due to errors`, false);
        }

        show("pred-result");
        window.lastPredictionData     = data;
        window.lastPredictionSelected = selected.filter(t => data.predicted[t]);
        renderPrediction(data, window.lastPredictionSelected, symbol);
        fetchNews(symbol, "pred-news");

    } catch (e) {
        clearInterval(progInterval);
        hide("pred-loader");
        document.getElementById("pred-btn").disabled = false;
        fill.style.width = "0%";
        toast("❌ " + e.message, false);
    }
}

/* ── RENDER PREDICTION ─────────────────────────────────── */
function renderPrediction(data, selected, symbol) {
    if (!selected || selected.length === 0) { toast("No model results to display", false); return; }
    const stock = stocks.find(s => s.symbol === symbol) || { name: symbol };

    const cMap = {
        ALL_UP:        { text: "🧠 All models agree — UPWARD TREND 📈",   cls: "up"    },
        ALL_DOWN:      { text: "🧠 All models agree — DOWNWARD TREND 📉", cls: "down"  },
        MAJORITY_UP:   { text: "🧠 Majority predict UPWARD TREND 📈",     cls: "up"    },
        MAJORITY_DOWN: { text: "🧠 Majority predict DOWNWARD TREND 📉",   cls: "down"  },
        SPLIT:         { text: "🧠 Models are split — No clear trend",     cls: "split" },
    };
    const c = cMap[data.consensus] || { text: "—", cls: "split" };
    const banner = document.getElementById("consensus-banner");
    banner.className = "consensus-banner " + c.cls;
    banner.textContent = c.text;

    const lb = document.getElementById("leaderboard");
    lb.innerHTML = `<div class="lb-row header"><span>ALGO</span><span>MAE</span><span>RMSE</span><span>DIR ACC</span><span></span></div>`;
    selected.forEach(tag => {
        if (!data.accuracy[tag]) return;
        const m   = data.accuracy[tag];
        const cls = tag === data.best ? " best" : "";
        lb.insertAdjacentHTML("beforeend", `
            <div class="lb-row${cls}" style="border-left:3px solid ${COLORS[tag]}">
                <span class="lb-tag" style="color:${COLORS[tag]}">${tag}</span>
                <span>${m.mae}</span><span>${m.rmse}</span><span>${m.da}%</span>
                <span class="lb-badge">${tag === data.best ? "🏆 BEST" : ""}</span>
            </div>`);
    });

    Object.values(predCharts).forEach(c => c && c.remove());
    predCharts = {};
    if (predActChart) { predActChart.remove(); predActChart = null; }

    const chartsWrap = document.getElementById("pred-charts");
    chartsWrap.innerHTML = "";

    const actPanel = document.createElement("div");
    actPanel.className = "pred-chart-panel";
    actPanel.innerHTML = `
        <div class="pred-chart-title"><span>📊 Actual — ${stock.name} (Last 20 Days)</span></div>
        <div id="pact-chart" class="pred-chart-box"></div>`;
    chartsWrap.appendChild(actPanel);

    setTimeout(() => {
        predActChart = LightweightCharts.createChart(document.getElementById("pact-chart"), chartOptions("pact-chart", 260));
        const cs = predActChart.addCandlestickSeries({
            upColor: "#00ff88", downColor: "#ff4455",
            borderUpColor: "#00ff88", borderDownColor: "#ff4455",
            wickUpColor: "#00ff88", wickDownColor: "#ff4455"
        });
        cs.setData(data.actual.map(c => ({ time: c.date, open: c.open, high: c.high, low: c.low, close: c.close })));
        predActChart.timeScale().fitContent();
    }, 50);

    selected.forEach((tag, idx) => {
        if (!data.predicted[tag]) return;
        const rows  = data.predicted[tag];
        const acc   = data.accuracy[tag];
        const color = COLORS[tag];
        const lastActual = data.actual[data.actual.length - 1].close;
        const lastPred   = rows[rows.length - 1].close;
        const chg        = lastPred - lastActual;
        const arrow = chg >= 0 ? "▲" : "▼";
        const cls   = chg >= 0 ? "var(--green)" : "var(--red)";
        const panel = document.createElement("div");
        panel.className = "pred-chart-panel";
        panel.style.borderTopColor = color;
        panel.innerHTML = `
            <div class="pred-chart-title" style="color:${color}">
                <span>${tag} — Next 15 Days</span>
                <span style="font-size:10px;color:var(--muted)">RMSE:${acc.rmse} MAE:${acc.mae} DA:${acc.da}%</span>
            </div>
            <div id="pc-${tag}" class="pred-chart-box"></div>
            <div style="margin-top:8px;font-size:12px;color:${cls}">${arrow} ${chg>=0?"+":""}${chg.toFixed(2)} forecast change</div>`;
        chartsWrap.appendChild(panel);
        setTimeout(() => {
            const chart = LightweightCharts.createChart(document.getElementById(`pc-${tag}`), chartOptions(`pc-${tag}`, 260));
            const cs = chart.addCandlestickSeries({
                upColor: color, downColor: "#ff4455",
                borderUpColor: color, borderDownColor: "#ff4455",
                wickUpColor: color, wickDownColor: "#ff4455"
            });
            cs.setData(rows.map(r => ({ time: r.date, open: r.open, high: r.high, low: r.low, close: r.close })));
            chart.timeScale().fitContent();
            predCharts[tag] = chart;
        }, 60 + idx * 40);
    });

    const tablesWrap = document.getElementById("pred-tables");
    tablesWrap.innerHTML = "";
    selected.forEach(tag => {
        if (!data.predicted[tag]) return;
        const rows  = data.predicted[tag];
        const color = COLORS[tag];
        const panel = document.createElement("div");
        panel.className = "pred-table-panel";
        panel.style.borderTop = `3px solid ${color}`;
        panel.innerHTML = `
            <div class="pred-table-title" style="color:${color}">${tag} — Predicted Prices</div>
            <div class="data-table-wrap"><table class="data-table">
                <thead><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th></tr></thead>
                <tbody>${rows.map(r => {
                    const chg = r.close - r.open;
                    return `<tr><td>${r.date}</td><td>${r.open.toFixed(2)}</td><td>${r.high.toFixed(2)}</td>
                        <td>${r.low.toFixed(2)}</td><td class="${chg>=0?"up-val":"down-val"}">${r.close.toFixed(2)}</td>
                        <td>${r.volume.toLocaleString()}</td></tr>`;
                }).join("")}</tbody>
            </table></div>`;
        tablesWrap.appendChild(panel);
    });
}

/* ── DELETE ────────────────────────────────────────────── */
async function deleteStock() {
    const symbol = selectedSymbols.del;
    const status = document.getElementById("del-status");
    if (!symbol) { setStatus(status, "⚠ Please select a stock first.", false); return; }
    if (!confirm(`Delete ${symbol} from your watchlist?`)) return;
    try {
        const res  = await fetch(`/api/stocks/${symbol}`, { method: "DELETE" });
        const data = await res.json();
        if (data.success) {
            toast(`${symbol} deleted`, true);
            setStatus(status, `✅ ${symbol} deleted.`, true);
            clearSearchSelect("del");
            await loadStocks();
            populateSelects();
        }
    } catch (e) {
        setStatus(status, "❌ Failed to delete. Please try again.", false);
    }
}

async function confirmDeleteAll() {
    const status = document.getElementById("del-all-status");
    if (!confirm("⚠ Delete ALL stocks permanently? This cannot be undone.")) return;
    if (!confirm("Are you absolutely sure?")) return;
    try {
        const res  = await fetch("/api/stocks/all", { method: "DELETE" });
        const data = await res.json();
        if (data.success) {
            toast("All stocks deleted", true);
            setStatus(status, "✅ All stocks deleted.", true);
            await loadStocks();
            populateSelects();
            document.getElementById("stat-total-val").textContent = "0";
        }
    } catch (e) {
        setStatus(status, "❌ Failed to delete. Please try again.", false);
    }
}

/* ── CHART OPTIONS ─────────────────────────────────────── */
function chartOptions(containerId, height) {
    return {
        autoSize: true,
        layout: { background: { color: "#0e0e1c" }, textColor: "#6a6a8a" },
        grid: { vertLines: { color: "rgba(255,255,255,0.04)" }, horzLines: { color: "rgba(255,255,255,0.04)" } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: "rgba(255,255,255,0.07)" },
        timeScale: { borderColor: "rgba(255,255,255,0.07)", timeVisible: true }
    };
}

/* ── UTILS ─────────────────────────────────────────────── */
function show(id) { const el = document.getElementById(id); if (el) el.classList.remove("hidden"); }
function hide(id) { const el = document.getElementById(id); if (el) el.classList.add("hidden"); }
function setStatus(el, msg, ok) { el.textContent = msg; el.className = "form-status " + (ok ? "ok" : "err"); }
function toast(msg, ok) {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.className = "toast show " + (ok ? "ok" : "err");
    setTimeout(() => t.classList.remove("show"), 4000);
}

/* ── EXPORT CSV ────────────────────────────────────────── */
function exportPredictionsCSV() {
    const data = window.lastPredictionData;
    const selected = window.lastPredictionSelected;
    if (!data || !selected) { toast("No predictions to export", false); return; }
    let csv = "Algo,Date,Open,High,Low,Close,Volume\n";
    selected.forEach(tag => {
        if (!data.predicted[tag]) return;
        data.predicted[tag].forEach(r => {
            csv += `${tag},${r.date},${r.open.toFixed(2)},${r.high.toFixed(2)},${r.low.toFixed(2)},${r.close.toFixed(2)},${r.volume}\n`;
        });
    });
    const blob = new Blob([csv], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = "predictions.csv";
    document.body.appendChild(link); link.click();
    document.body.removeChild(link); URL.revokeObjectURL(url);
}

function getFaceSVG(type) {

    let bg = "#ffd93d"; // yellow
    let mouth = `<line x1="10" y1="24" x2="30" y2="24" stroke="black" stroke-width="2"/>`; // neutral

    if (type === "positive") {
        bg = "#00ff88";
        mouth = `<path d="M10 24 Q20 32 30 24" stroke="black" stroke-width="2" fill="none"/>`;
    } 
    else if (type === "negative") {
        bg = "#ff4455";
        mouth = `<path d="M10 28 Q20 18 30 28" stroke="black" stroke-width="2" fill="none"/>`;
    }

    return `
    <div style="
        position:absolute;
        top:8px;
        right:8px;
        width:36px;
        height:36px;
        border-radius:12px;
        background:${bg};
        display:flex;
        align-items:center;
        justify-content:center;
        box-shadow:0 0 8px ${bg}66;
    ">
        <svg width="26" height="26" viewBox="0 0 40 40">
            <!-- eyes -->
            <circle cx="14" cy="14" r="5" fill="white"/>
            <circle cx="26" cy="14" r="5" fill="white"/>
            <circle cx="14" cy="14" r="2.5" fill="black"/>
            <circle cx="26" cy="14" r="2.5" fill="black"/>

            <!-- mouth -->
            ${mouth}
        </svg>
    </div>`;
}
/* ── NEWS MODULE ───────────────────────────────────────── */
async function fetchNews(symbol, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `<div class="spinner"></div><span style="margin-left:8px;font-size:12px;color:var(--muted)">Loading news...</span>`;

    try {
        const res  = await fetch(`/api/news/${symbol}`);
        const news = await res.json();

        if (news.error || news.length === 0) {
            container.innerHTML = `<p style="font-size:12px;color:var(--muted)">No recent news found for ${symbol}.</p>`;
            return;
        }

        let html = '<div style="display:flex;flex-direction:column;gap:12px;margin-top:14px;">';

        news.forEach(n => {
            const face = getFaceSVG(n.sentiment);
            const d = n.pubDate ? new Date(n.pubDate).toLocaleDateString() : "Recent";

            html += `
            <div style="position:relative;background:var(--bg3);padding:14px 54px 14px 14px;border-radius:8px;border:1px solid var(--border);">

                ${face}

                <a href="${n.link}" target="_blank" style="color:var(--accent);text-decoration:none;font-size:13px;font-weight:bold;margin-bottom:6px;display:block;">
                    ${n.title}
                </a>

                <p style="color:var(--text);font-size:11px;line-height:1.5;">
                    ${n.summary}
                </p>

                <div style="color:var(--muted);font-size:10px;margin-top:8px;padding-right:44px;">
                    ${d}
                </div>

            </div>`;
        });

        html += "</div>";
        container.innerHTML = html;

    } catch (e) {
        container.innerHTML = `<p style="font-size:12px;color:var(--red)">❌ Failed to load news for ${symbol}.</p>`;
    }
}

async function renderMarketNews() {
    const container = document.getElementById("market-news-feed");
    container.innerHTML = `<div class="loader"><div class="spinner"></div><span style="margin-left:8px;">Aggregating portfolio news...</span></div>`;
    if (!stocks || stocks.length === 0) {
        container.innerHTML = `<p style="color:var(--muted)">Add some stocks first to see news here.</p>`; return;
    }
    let allNews = [];
    for (const s of stocks.slice(0, 10)) {
        try {
            const res  = await fetch(`/api/news/${s.symbol}`);
            const news = await res.json();
            if (news && !news.error) {
                news.forEach(n => { n.stockSymbol = s.symbol; n.stockName = s.name; });
                allNews = allNews.concat(news);
            }
            await new Promise(r => setTimeout(r, 100));
        } catch (e) { console.error(`News fetch failed for ${s.symbol}:`, e); }
    }
    allNews.sort((a, b) => (!a.pubDate || !b.pubDate) ? 0 : new Date(b.pubDate) - new Date(a.pubDate));
    if (allNews.length === 0) {
        container.innerHTML = `<p style="color:var(--muted)">No news found across your portfolio right now.</p>`; return;
    }
    // Store all news globally so search can filter without re-fetching
    window.allPortfolioNews = allNews;
    renderNewsWithFilter("");
}

function renderNewsWithFilter(query) {
    const container = document.getElementById("market-news-feed");
    const allNews   = window.allPortfolioNews || [];

    const filtered = query
        ? allNews.filter(n =>
            n.stockSymbol.toLowerCase().includes(query.toLowerCase()) ||
            n.stockName.toLowerCase().includes(query.toLowerCase()))
        : allNews;

    let html = `
        <div style="margin-bottom:20px;position:relative;">
            <input 
                id="news-search-input"
                type="text"
                placeholder="🔍 Search by company or ticker…"
                oninput="renderNewsWithFilter(this.value)"
                value="${query}"
                style="width:100%;max-width:400px;background:var(--bg3);border:1px solid var(--border);
                       border-radius:6px;color:var(--text);font-family:'Space Mono',monospace;
                       font-size:13px;padding:11px 14px;outline:none;transition:border .2s;"
                onfocus="this.style.borderColor='var(--accent)'"
                onblur="this.style.borderColor='var(--border)'"
            />
            <span style="font-size:11px;color:var(--muted);margin-left:12px;">${filtered.length} article${filtered.length !== 1 ? "s" : ""}</span>
        </div>
        <div style="display:flex;flex-direction:column;gap:16px;max-width:800px;">
    `;

    if (filtered.length === 0) {
        html += `<p style="color:var(--muted)">No news found for "${query}".</p>`;
    } else {
        filtered.forEach(n => {
            const d = n.pubDate ? new Date(n.pubDate).toLocaleString() : "Recent";
            html += `<div style="position:relative;background:var(--bg2);padding:18px;border-radius:10px;border:1px solid var(--border);">
                ${getFaceSVG(n.sentiment)}
                <div style="margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;">
                    <span style="background:rgba(0,229,255,0.1);color:var(--accent);padding:6px 12px;border-radius:5px;font-size:12px;font-weight:bold;">${n.stockName} (${n.stockSymbol})</span>
                    <span style="color:var(--muted);font-size:12px;padding-right:48px;">${d}</span></div>
                <a href="${n.link}" target="_blank" style="color:#fff;text-decoration:none;font-size:17px;font-family:'Syne',sans-serif;font-weight:bold;margin-bottom:8px;display:block;">${n.title}</a>
                <p style="color:var(--text);font-size:13px;line-height:1.6;padding-right:44px;">${n.summary}</p></div>`;
        });
    }

    html += "</div>";
    container.innerHTML = html;

    // Keep focus in search box after re-render
    if (query) {
        const inp = document.getElementById("news-search-input");
        if (inp) { inp.focus(); inp.setSelectionRange(inp.value.length, inp.value.length); }
    }
}



    
