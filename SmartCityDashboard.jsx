import { useState, useEffect, useRef } from "react";

// ── Constants (mirrors Python backend) ───────────────────────────────────────
const MAX_CAPACITY = {
  electricity_load_mw: 2000,
  daily_water_release_mld: 500,
  daily_admissions: 300,
};
const EXPECTED_SUPPLY = {
  electricity_load_mw: 1800,
  daily_water_release_mld: 450,
  daily_admissions: 250,
};
const RESOURCE_META = {
  electricity_load_mw:     { label: "Electricity",        unit: "MW",  icon: "⚡", color: "#f0c040", glow: "#f0c04060" },
  daily_water_release_mld: { label: "Water",              unit: "MLD", icon: "💧", color: "#38d9f5", glow: "#38d9f560" },
  daily_admissions:        { label: "Hospital Admits",    unit: "pts", icon: "🏥", color: "#c084fc", glow: "#c084fc60" },
};
const AREAS = ["Balewadi", "Shivajinagar", "Kothrud", "Hadapsar", "Wakad"];

// ── Data helpers ─────────────────────────────────────────────────────────────
function generateData(n = 30) {
  const now = Date.now();
  return Array.from({ length: n }, (_, i) => ({
    timestamp: new Date(now - (n - 1 - i) * 3_600_000),
    area: AREAS[i % AREAS.length],
    electricity_load_mw:     1200 + Math.random() * 700,
    daily_water_release_mld: 280  + Math.random() * 200,
    daily_admissions:        130  + Math.random() * 140,
  }));
}

function addLiveRow(data) {
  const last = data[data.length - 1];
  const next = {
    timestamp: new Date(),
    area: AREAS[Math.floor(Math.random() * AREAS.length)],
    electricity_load_mw:     Math.max(800, last.electricity_load_mw     + (Math.random() - 0.48) * 80),
    daily_water_release_mld: Math.max(200, last.daily_water_release_mld + (Math.random() - 0.48) * 25),
    daily_admissions:        Math.max(80,  last.daily_admissions         + (Math.random() - 0.48) * 15),
  };
  return [...data.slice(1), next];
}

function getStatus(pct) {
  if (pct > 90) return { label: "OVERLOAD",    color: "#ef4444" };
  if (pct > 70) return { label: "HIGH STRAIN", color: "#fb923c" };
  return               { label: "NOMINAL",     color: "#4ade80" };
}

// ── Sparkline ─────────────────────────────────────────────────────────────────
function Sparkline({ values, color, height = 48, width = 200 }) {
  if (!values?.length) return null;
  const mn = Math.min(...values), mx = Math.max(...values);
  const rng = mx - mn || 1;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - mn) / rng) * (height - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const gradId = `sg-${color.replace("#", "")}`;
  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ filter: `drop-shadow(0 0 3px ${color})` }}
      />
      {/* Glowing last dot */}
      {(() => {
        const lv = values[values.length - 1];
        const lx = width;
        const ly = height - ((lv - mn) / rng) * (height - 4) - 2;
        return (
          <circle cx={lx} cy={ly} r="3" fill={color}
            style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
        );
      })()}
    </svg>
  );
}

// ── Zone bar ─────────────────────────────────────────────────────────────────
function ZoneBar({ area, value, max, color }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
      <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 10, color: "#7c8fa6", width: 90, flexShrink: 0 }}>
        {area.toUpperCase()}
      </span>
      <div style={{ flex: 1, height: 3, background: "#1a2234", borderRadius: 2, position: "relative", overflow: "hidden" }}>
        <div style={{
          position: "absolute", inset: 0, width: `${pct}%`,
          background: `linear-gradient(90deg, ${color}66, ${color})`,
          borderRadius: 2,
          boxShadow: `0 0 8px ${color}80`,
          transition: "width 0.8s ease",
        }} />
      </div>
      <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 10, color, width: 52, textAlign: "right" }}>
        {value.toFixed(0)}
      </span>
    </div>
  );
}

// ── Resource Card ─────────────────────────────────────────────────────────────
function ResourceCard({ resourceKey, data, visible }) {
  const meta    = RESOURCE_META[resourceKey];
  const maxCap  = MAX_CAPACITY[resourceKey];
  const supply  = EXPECTED_SUPPLY[resourceKey];
  const latest  = data[data.length - 1]?.[resourceKey] ?? 0;
  const usage   = (latest / maxCap) * 100;
  const status  = getStatus(usage);
  const peak    = Math.max(...data.map(d => d[resourceKey]));
  const gap     = supply - latest;
  const sparkVals = data.map(d => d[resourceKey]);

  const zoneAvgs = AREAS.map(area => {
    const rows = data.filter(d => d.area === area);
    const avg = rows.length ? rows.reduce((s, r) => s + r[resourceKey], 0) / rows.length : 0;
    return { area, avg };
  });

  const alerts = [];
  if (latest > maxCap)         alerts.push({ msg: "CAPACITY EXCEEDED",    color: "#ef4444" });
  else if (latest > .9*maxCap) alerts.push({ msg: "HIGH DEMAND ALERT",    color: "#fb923c" });
  if (latest < .5*maxCap)      alerts.push({ msg: "UNDERUTILIZATION",     color: "#38d9f5" });

  return (
    <div style={{
      background: "rgba(8,14,26,0.92)",
      border: `1px solid ${meta.color}22`,
      borderRadius: 20,
      padding: "28px 24px",
      display: "flex",
      flexDirection: "column",
      gap: 20,
      position: "relative",
      overflow: "hidden",
      opacity: visible ? 1 : 0,
      transform: visible ? "translateY(0)" : "translateY(24px)",
      transition: "opacity 0.6s ease, transform 0.6s ease",
      backdropFilter: "blur(20px)",
    }}>
      {/* Top glow line */}
      <div style={{
        position: "absolute", top: 0, left: "10%", right: "10%", height: 1,
        background: `linear-gradient(90deg, transparent, ${meta.color}, transparent)`,
        boxShadow: `0 0 16px ${meta.color}`,
      }} />
      {/* Corner accent */}
      <div style={{
        position: "absolute", top: 16, right: 16,
        width: 6, height: 6, borderRadius: "50%",
        background: status.color,
        boxShadow: `0 0 10px ${status.color}, 0 0 20px ${status.color}`,
        animation: "pulse 2s ease-in-out infinite",
      }} />

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontSize: 11, letterSpacing: "0.2em", color: "#445566", fontFamily: "'Space Mono', monospace", marginBottom: 6 }}>
            {meta.icon} {meta.label.toUpperCase()}
          </div>
          <div style={{
            fontSize: 38, fontWeight: 900, color: meta.color,
            fontFamily: "'Space Mono', monospace", lineHeight: 1,
            textShadow: `0 0 30px ${meta.glow}`,
          }}>
            {latest.toFixed(0)}
            <span style={{ fontSize: 14, color: "#445566", marginLeft: 6 }}>{meta.unit}</span>
          </div>
        </div>
        <div style={{
          padding: "4px 12px", borderRadius: 30, fontSize: 10, fontWeight: 700,
          background: `${status.color}18`, color: status.color,
          border: `1px solid ${status.color}40`, fontFamily: "'Space Mono', monospace",
          letterSpacing: "0.1em",
        }}>
          {status.label}
        </div>
      </div>

      {/* Sparkline */}
      <div>
        <div style={{ fontSize: 9, color: "#334455", fontFamily: "'Space Mono', monospace", letterSpacing: "0.15em", marginBottom: 8 }}>
          ▸ LAST 30 READINGS
        </div>
        <Sparkline values={sparkVals} color={meta.color} width={260} height={52} />
      </div>

      {/* Capacity bar */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "'Space Mono', monospace", fontSize: 10, color: "#445566", marginBottom: 6 }}>
          <span>INFRASTRUCTURE LOAD</span>
          <span style={{ color: status.color }}>{usage.toFixed(1)}%</span>
        </div>
        <div style={{ height: 4, background: "#0d1420", borderRadius: 2 }}>
          <div style={{
            height: "100%", width: `${Math.min(usage, 100)}%`,
            background: `linear-gradient(90deg, ${meta.color}55, ${meta.color})`,
            borderRadius: 2,
            boxShadow: `0 0 12px ${meta.color}80`,
            transition: "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
          }} />
        </div>
      </div>

      {/* Stats grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        {[
          ["PEAK", `${peak.toFixed(0)} ${meta.unit}`],
          ["TARGET", `${supply} ${meta.unit}`],
          [gap >= 0 ? "SURPLUS" : "DEFICIT", `${Math.abs(gap).toFixed(0)} ${meta.unit}`],
          ["MAX CAP", `${maxCap} ${meta.unit}`],
        ].map(([label, value]) => (
          <div key={label} style={{
            background: "#050c18", borderRadius: 10, padding: "10px 14px",
            border: "1px solid #0d1a2a",
          }}>
            <div style={{ fontSize: 9, color: "#334455", fontFamily: "'Space Mono', monospace", letterSpacing: "0.15em", marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: meta.color, fontFamily: "'Space Mono', monospace" }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Zone breakdown */}
      <div>
        <div style={{ fontSize: 9, color: "#334455", fontFamily: "'Space Mono', monospace", letterSpacing: "0.15em", marginBottom: 10 }}>▸ ZONE BREAKDOWN</div>
        {zoneAvgs.map(({ area, avg }) => (
          <ZoneBar key={area} area={area} value={avg} max={maxCap} color={meta.color} />
        ))}
      </div>

      {/* Alerts */}
      {alerts.map(({ msg, color }, i) => (
        <div key={i} style={{
          padding: "8px 14px", borderRadius: 8, fontSize: 10,
          background: `${color}12`, border: `1px solid ${color}35`, color,
          fontFamily: "'Space Mono', monospace", letterSpacing: "0.1em",
          animation: "blink 1.5s ease-in-out infinite",
        }}>
          ◈ {msg}
        </div>
      ))}
    </div>
  );
}

// ── Metric ticker ─────────────────────────────────────────────────────────────
function TickerItem({ label, value, unit, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "0 28px", borderRight: "1px solid #0d1a2a", flexShrink: 0 }}>
      <span style={{ fontSize: 10, color: "#445566", fontFamily: "'Space Mono', monospace", letterSpacing: "0.1em" }}>{label}</span>
      <span style={{ fontSize: 14, fontWeight: 700, color, fontFamily: "'Space Mono', monospace", textShadow: `0 0 12px ${color}` }}>
        {value} <span style={{ fontSize: 10, color: "#445566" }}>{unit}</span>
      </span>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function SmartCityDashboard() {
  const [data, setData]       = useState(generateData);
  const [tick, setTick]       = useState(0);
  const [visible, setVisible] = useState(false);
  const [apiMode, setApiMode] = useState(false);
  const [apiUrl]              = useState("http://localhost:5000/api/dashboard");

  // Reveal animation
  useEffect(() => { setTimeout(() => setVisible(true), 100); }, []);

  // Live tick — optionally fetch from Flask API
  useEffect(() => {
    const id = setInterval(async () => {
      if (apiMode) {
        try {
          const res = await fetch(apiUrl);
          if (res.ok) {
            // API returns summary; for demo we still animate locally
          }
        } catch (_) {}
      }
      setData(prev => addLiveRow(prev));
      setTick(t => t + 1);
    }, 2000);
    return () => clearInterval(id);
  }, [apiMode, apiUrl]);

  const now    = data[data.length - 1]?.timestamp;
  const latest = (key) => data[data.length - 1]?.[key] ?? 0;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#03070f",
      fontFamily: "'Space Mono', monospace",
      color: "#c8d8e8",
      overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@800;900&display=swap');
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.3)} }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.6} }
        @keyframes scan {
          0%   { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
        @keyframes marqueeTicker {
          0%   { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #03070f; }
        ::-webkit-scrollbar-thumb { background: #1a2a3a; border-radius: 2px; }
      `}</style>

      {/* Scanline overlay */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none", zIndex: 100,
        background: "repeating-linear-gradient(to bottom, transparent 0px, transparent 2px, rgba(0,0,0,0.04) 2px, rgba(0,0,0,0.04) 4px)",
      }} />
      {/* Scan beam */}
      <div style={{
        position: "fixed", left: 0, right: 0, height: "2px", zIndex: 99, pointerEvents: "none",
        background: "linear-gradient(90deg, transparent, #38d9f510, transparent)",
        animation: "scan 8s linear infinite",
      }} />

      {/* Header */}
      <header style={{
        padding: "20px 36px",
        borderBottom: "1px solid #0d1a2a",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "rgba(3,7,15,0.9)",
        backdropFilter: "blur(20px)",
        position: "sticky", top: 0, zIndex: 50,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <div style={{
            width: 36, height: 36, borderRadius: "50%",
            background: "conic-gradient(#38d9f5, #c084fc, #f0c040, #38d9f5)",
            animation: "pulse 3s ease-in-out infinite",
            boxShadow: "0 0 20px #38d9f540",
          }} />
          <div>
            <div style={{ fontSize: 10, letterSpacing: "0.3em", color: "#334455" }}>URBAN INTELLIGENCE PLATFORM</div>
            <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 22, fontWeight: 900, letterSpacing: "-0.02em", color: "#c8d8e8" }}>
              SMART CITY <span style={{ color: "#38d9f5", textShadow: "0 0 20px #38d9f5" }}>NEXUS</span>
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 9, color: "#334455", letterSpacing: "0.2em" }}>
              LIVE · TICK {tick.toString().padStart(4, "0")}
            </div>
            <div style={{ fontSize: 11, color: "#445566" }}>
              {now?.toLocaleTimeString("en-IN", { hour12: false })}
            </div>
          </div>
          <div style={{
            width: 8, height: 8, borderRadius: "50%", background: "#4ade80",
            boxShadow: "0 0 10px #4ade80, 0 0 20px #4ade80",
            animation: "pulse 1s ease-in-out infinite",
          }} />
        </div>
      </header>

      {/* Ticker bar */}
      <div style={{
        borderBottom: "1px solid #0d1a2a",
        background: "#040a14",
        overflow: "hidden", height: 42,
        display: "flex", alignItems: "center",
      }}>
        <div style={{
          display: "flex",
          animation: "marqueeTicker 20s linear infinite",
          width: "200%",
        }}>
          {[...Array(2)].map((_, ri) => (
            <div key={ri} style={{ display: "flex", flexShrink: 0, width: "50%" }}>
              <TickerItem label="ELECTRICITY" value={latest("electricity_load_mw").toFixed(0)} unit="MW" color="#f0c040" />
              <TickerItem label="WATER" value={latest("daily_water_release_mld").toFixed(0)} unit="MLD" color="#38d9f5" />
              <TickerItem label="ADMISSIONS" value={latest("daily_admissions").toFixed(0)} unit="pts" color="#c084fc" />
              <TickerItem label="ELEC CAPACITY" value={((latest("electricity_load_mw")/2000)*100).toFixed(1)} unit="%" color="#f0c040" />
              <TickerItem label="WATER CAPACITY" value={((latest("daily_water_release_mld")/500)*100).toFixed(1)} unit="%" color="#38d9f5" />
              <TickerItem label="HOSP CAPACITY" value={((latest("daily_admissions")/300)*100).toFixed(1)} unit="%" color="#c084fc" />
            </div>
          ))}
        </div>
      </div>

      {/* Cards */}
      <main style={{ padding: "32px 36px", maxWidth: 1200, margin: "0 auto" }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: 24,
        }}>
          {Object.keys(RESOURCE_META).map((key, i) => (
            <div key={key} style={{ transitionDelay: `${i * 120}ms` }}>
              <ResourceCard resourceKey={key} data={data} visible={visible} />
            </div>
          ))}
        </div>

        {/* Footer */}
        <div style={{
          marginTop: 36, textAlign: "center",
          fontSize: 9, color: "#1a2a3a", letterSpacing: "0.3em",
        }}>
          SMART CITY INTELLIGENCE SYSTEM · PUNE, MH · AUTO-REFRESH 2s
          {" · "}
          <span style={{ color: "#334455" }}>
            BACKEND: <span style={{ color: "#38d9f580" }}>python app.py</span> → {apiUrl}
          </span>
        </div>
      </main>
    </div>
  );
}
