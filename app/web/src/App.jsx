import React, { useEffect, useMemo, useState } from "react";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

const C = { navy: "#14213D", orange: "#B4530A", amber: "#F4A259", blue: "#1A5FD0", grey: "#8FA3C0", line: "#D9D6CC", ink2: "#4A5568" };
const STAGE = {
  initial_DoE: { label: "Plan initial (32)", color: C.grey },
  active_learning: { label: "Active learning (15)", color: C.orange },
  bayesian_opt: { label: "Optim. bayésienne (4)", color: C.blue },
};
const STRAT = {
  random: "Hasard",
  greedy_xgb: "Top 5 XGBoost (papier)",
  greedy_rf: "Top 5 Random Forest",
  gp_mean: "GP, moyenne seule",
  gp_ei: "GP + Expected Improvement",
  gp_ucb: "GP + UCB",
};
const STRAT_COLOR = { random: C.grey, greedy_xgb: C.orange, greedy_rf: C.amber, gp_mean: "#7FA6E8", gp_ei: C.blue, gp_ucb: C.navy };
const ORACLE = { gp51: "GP · 51 mesures", rf51: "Random Forest · 51 mesures", xgb51: "XGBoost · 51 mesures", gp32: "GP · 32 essais initiaux" };
const fmt = (x, d = 1) => (x == null ? "–" : x.toFixed(d).replace(".", ","));
const pct = (x) => (x < 0.01 ? "< 1 %" : `${Math.round(x * 100)} %`);

const baseLayout = (extra = {}) => ({
  font: { family: "IBM Plex Sans, Arial", size: 13, color: C.navy },
  paper_bgcolor: "#fff", plot_bgcolor: "#fff",
  margin: { l: 56, r: 16, t: 36, b: 52 },
  hoverlabel: { font: { family: "IBM Plex Sans" } },
  legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom" },
  ...extra,
});
const cfg = { displayModeBar: false, responsive: true };

function useApi(url) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    if (!url) return;
    let alive = true;
    setError(null);
    fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((d) => alive && setData(d)).catch((e) => alive && setError(String(e)));
    return () => { alive = false; };
  }, [url]);
  return [data, error];
}

const Loading = ({ error }) => <div className="loading">{error ? `Erreur : ${error}` : "Calcul en cours…"}</div>;

function Seg({ options, value, onChange }) {
  return (
    <div className="seg">
      {Object.entries(options).map(([k, v]) => (
        <button key={k} className={String(value) === String(k) ? "on" : ""} onClick={() => onChange(k)}>{v}</button>
      ))}
    </div>
  );
}

function Stat({ num, children }) {
  return <div className="stat"><span className="num">{num}</span><span className="small">{children}</span></div>;
}

function Recipe({ p }) {
  return (
    <dl className="recipe">
      <dt>Résine époxy</dt><dd>{p.MW_epoxy}</dd>
      <dt>Durcisseur</dt><dd>{p.MW_curing_agent}</dd>
      <dt>Ratio r</dt><dd>{fmt(p.r, 2)}</dd>
      <dt>Cuisson</dt><dd>{p.T_cure} °C</dd>
    </dl>
  );
}

/* ------------------------------------------------------------------ 1. Campaign */
function Campaign() {
  const [d, err] = useApi("/api/campaign");
  const [sel, setSel] = useState(17);
  if (!d) return <Loading error={err} />;
  const rows = d.rows;
  const picked = rows.find((r) => r.id === sel);
  const traces = Object.entries(STAGE).map(([s, st]) => {
    const g = rows.filter((r) => r.stage === s);
    return {
      type: "scatter", mode: "markers", name: st.label, x: g.map((r) => r.id), y: g.map((r) => r.sigma_ad_MPa),
      error_y: { type: "data", array: g.map((r) => r.sigma_std_MPa || 0), color: st.color, thickness: 1.2, width: 3 },
      marker: { color: st.color, size: g.map((r) => (r.id === sel ? 16 : 10)), line: { color: "#fff", width: 1 } },
      customdata: g.map((r) => [r.MW_epoxy, r.MW_curing_agent, r.r, r.T_cure]),
      hovertemplate: "n°%{x} · %{y:.1f} MPa<br>résine %{customdata[0]}, durcisseur %{customdata[1]}<br>r = %{customdata[2]}, %{customdata[3]} °C<extra></extra>",
    };
  });
  const count = (key, vals) => Object.entries(STAGE).slice(0, 2).map(([s, st]) => ({
    type: "bar", name: st.label, marker: { color: st.color }, x: vals.map(String),
    y: vals.map((v) => rows.filter((r) => r.stage === s && r[key] === v).length),
  }));
  return (
    <div className="page">
      <aside className="side">
        <div className="card">
          <h3>La question</h3>
          <p className="question">La boucle d’active learning a-t-elle trouvé de meilleurs adhésifs ?</p>
          <p className="muted small" style={{ marginTop: 8 }}>Dans le papier, 3 cycles de 5 essais sont choisis parmi les recettes que l’XGBoost prédit les meilleures, puis 4 essais d’optimisation bayésienne affinent r et T.</p>
        </div>
        <div className="card">
          <h3>Ce qu’on observe</h3>
          <Stat num={`${d.al_beating_initial} / ${d.n_al}`}>propositions qui battent le meilleur essai du plan initial (n°17, {fmt(d.best_initial)} MPa)</Stat>
          <Stat num={`${d.al_with_mwc400} / ${d.n_al}`}>propositions avec le durcisseur 400 ; aucune ne revient près du n°17</Stat>
          <Stat num="35,8">MPa, le record, trouvé pendant l’optimisation bayésienne</Stat>
        </div>
        {picked && (
          <div className="card">
            <h3>Essai n°{picked.id} · {fmt(picked.sigma_ad_MPa)} MPa</h3>
            <Recipe p={picked} />
            {picked.appearance && <p className="muted small" style={{ marginTop: 6 }}>Aspect : {picked.appearance}</p>}
            <p className="muted small" style={{ marginTop: 6 }}>Cliquez sur un point pour voir sa recette.</p>
          </div>
        )}
        <div className="verdict">
          <h3>Lecture</h3>
          <p>La boucle « top 5 prédit » a rendu le modèle plus précis dans la zone forte, mais n’a pas amélioré le meilleur adhésif. Améliorer la prédiction et optimiser sont deux objectifs différents.</p>
        </div>
      </aside>
      <section className="main">
        <div className="panel">
          <h2>Les 51 essais dans l’ordre</h2>
          <p className="sub">Barres : écart-type annoncé entre éprouvettes. Ligne pointillée : meilleur essai du plan initial.</p>
          <Plot data={traces} config={cfg} style={{ width: "100%", height: 420 }} useResizeHandler
            onClick={(e) => setSel(e.points[0].x)}
            layout={baseLayout({
              xaxis: { title: "n° d’essai", zeroline: false, gridcolor: "#eee" },
              yaxis: { title: "résistance mesurée (MPa)", range: [0, 38], gridcolor: "#eee" },
              shapes: [{ type: "line", x0: 0, x1: 52, y0: d.best_initial, y1: d.best_initial, line: { dash: "dot", color: C.navy, width: 1.5 } }],
              annotations: [{ x: 17, y: d.best_initial, text: "n°17", showarrow: true, ay: -30, font: { color: C.navy } }],
            })} />
        </div>
        <div className="grid2">
          <div className="panel">
            <h2>Durcisseur choisi</h2>
            <p className="sub">Le n°17 utilise le durcisseur 230.</p>
            <Plot data={count("MW_curing_agent", [230, 400, 2000, 4000])} config={cfg} useResizeHandler style={{ width: "100%", height: 280 }}
              layout={baseLayout({ barmode: "group", xaxis: { title: "masse du durcisseur", type: "category" }, yaxis: { title: "nombre d’essais" } })} />
          </div>
          <div className="panel">
            <h2>Température de cuisson</h2>
            <p className="sub">Le n°17 est cuit à 130 °C.</p>
            <Plot data={count("T_cure", [90, 130, 170, 210])} config={cfg} useResizeHandler style={{ width: "100%", height: 280 }}
              layout={baseLayout({ barmode: "group", xaxis: { title: "température (°C)", type: "category" }, yaxis: { title: "nombre d’essais" } })} />
          </div>
        </div>
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ 2. R² */
function R2() {
  const [model, setModel] = useState("xgb");
  const [depth, setDepth] = useState(5);
  const [paperHp, setPaperHp] = useState(false);
  const url = `/api/loocv?model=${model}${model !== "enet" ? `&max_depth=${depth}` : ""}${model === "xgb" && paperHp ? "&paper_hp=true" : ""}`;
  const [d, err] = useApi(url);
  const curve = d?.curve;
  return (
    <div className="page">
      <aside className="side">
        <div className="card">
          <h3>La question</h3>
          <p className="question">La hausse du R² traduit-elle un meilleur modèle, ou un jeu évalué qui change ?</p>
          <p className="muted small" style={{ marginTop: 8 }}>Le papier annonce un R² en leave-one-out qui passe de {fmt(0.68, 2)} à {fmt(0.85, 2)}. Mais chaque essai ajouté devient aussi un point de test.</p>
        </div>
        <div className="card controls">
          <h3>Réglages</h3>
          <Seg options={{ xgb: "XGBoost", rf: "Random Forest", enet: "Elastic Net" }} value={model} onChange={setModel} />
          {model !== "enet" && !(model === "xgb" && paperHp) && (
            <label>Profondeur max des arbres : {depth}
              <input type="range" min={2} max={8} value={depth} onChange={(e) => setDepth(+e.target.value)} />
            </label>
          )}
          {model === "xgb" && (
            <label className="check"><input type="checkbox" checked={paperHp} onChange={(e) => setPaperHp(e.target.checked)} />Hyperparamètres du papier à chaque cycle (tableau S4)</label>
          )}
        </div>
        {curve && (
          <div className="card">
            <h3>Ce qu’on observe</h3>
            <Stat num={`+${fmt(curve[3].r2_all - curve[0].r2_all, 2)}`}>R² sur tout le jeu ({fmt(curve[0].r2_all, 2)} → {fmt(curve[3].r2_all, 2)}), comme dans le papier</Stat>
            <Stat num={`${curve[3].r2_doe - curve[0].r2_doe >= 0 ? "+" : ""}${fmt(curve[3].r2_doe - curve[0].r2_doe, 2)}`}>R² sur les 32 essais du plan initial, qui couvrent tout l’espace</Stat>
          </div>
        )}
        <div className="verdict">
          <h3>Lecture</h3>
          <p>Les 15 essais ajoutés sont presque identiques et tous dans la zone forte : chacun est facile à prédire grâce à ses voisins. Pour mesurer un vrai progrès, il faut un jeu de test figé.</p>
        </div>
      </aside>
      <section className="main">
        {!d ? <Loading error={err} /> : (
          <>
            <div className="panel">
              <h2>R² en leave-one-out selon la taille du jeu</h2>
              <p className="sub">Mêmes prédictions, deux façons de les noter.</p>
              <Plot config={cfg} useResizeHandler style={{ width: "100%", height: 360 }}
                data={[
                  { x: curve.map((c) => c.n), y: curve.map((c) => c.r2_all), name: "Sur tout le jeu (ce que rapporte le papier)", mode: "lines+markers", line: { color: C.orange, width: 3 }, marker: { size: 9 } },
                  { x: curve.map((c) => c.n), y: curve.map((c) => c.r2_doe), name: "Sur les 32 essais du plan initial", mode: "lines+markers", line: { color: C.blue, width: 3 }, marker: { size: 9 } },
                  { x: [32, 47], y: [0.68, 0.85], name: "Chiffres du papier", mode: "markers", marker: { symbol: "diamond-open", size: 12, color: C.navy, line: { width: 2 } } },
                ]}
                layout={baseLayout({ xaxis: { title: "nombre d’essais", tickvals: [32, 37, 42, 47] }, yaxis: { title: "R²", range: [0, 1], gridcolor: "#eee" } })} />
            </div>
            <div className="panel">
              <h2>Mesuré contre prédit, avec 47 essais</h2>
              <p className="sub">Chaque point est prédit par un modèle qui ne l’a pas vu. Les points orange, regroupés, sont faciles.</p>
              <Plot config={cfg} useResizeHandler style={{ width: "100%", height: 380 }}
                data={[
                  ...Object.entries(STAGE).slice(0, 2).map(([s, st]) => {
                    const g = d.predictions_47.filter((p) => p.stage === s);
                    return { type: "scatter", mode: "markers", name: st.label, x: g.map((p) => p.measured), y: g.map((p) => p.predicted), text: g.map((p) => `n°${p.id}`), marker: { color: st.color, size: 10 }, hovertemplate: "%{text}<br>mesuré %{x:.1f}, prédit %{y:.1f}<extra></extra>" };
                  }),
                  { x: [0, 35], y: [0, 35], mode: "lines", line: { color: C.line, dash: "dot" }, showlegend: false, hoverinfo: "skip" },
                ]}
                layout={baseLayout({ xaxis: { title: "mesuré (MPa)", range: [0, 35] }, yaxis: { title: "prédit (MPa)", range: [0, 35], gridcolor: "#eee" } })} />
            </div>
          </>
        )}
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ 3. Replay */
function Replay() {
  const [mwe, setMwe] = useState(370);
  const [mwc, setMwc] = useState(400);
  const [right, setRight] = useState("p_beat");
  const [d, err] = useApi(`/api/replay?mwe=${mwe}&mwc=${mwc}`);
  const RIGHT = { p_beat: "Chance de battre 31,9", gp_mean: "Prédiction du GP", gp_sd: "Incertitude du GP" };
  const heat = (z, title, colorscale, zmin, zmax, fmtStr) => ({
    type: "heatmap", x: d.r, y: d.T, z, colorscale, zmin, zmax, colorbar: { title: { text: title, side: "right" }, thickness: 12 },
    hovertemplate: `r = %{x}<br>T = %{y} °C<br>${title} : %{z:${fmtStr}}<extra></extra>`,
  });
  const bo = d?.bo_points || [];
  const boTrace = {
    type: "scatter", mode: "markers+text", x: bo.map((p) => p.r), y: bo.map((p) => p.T), text: bo.map((p) => fmt(p.measured)),
    textposition: bo.map((p, i) => (i % 2 ? "bottom center" : "top center")), textfont: { color: "#fff", size: 13, family: "Space Grotesk" },
    marker: { color: "#fff", size: 11, line: { color: C.navy, width: 2 } }, showlegend: false,
    hovertemplate: "proposition de la BO<br>mesuré %{text} MPa<extra></extra>",
  };
  const top = d?.plateaus?.[0];
  const rec = bo.find((p) => p.id === 51);
  return (
    <div className="page">
      <aside className="side">
        <div className="card">
          <h3>La question</h3>
          <p className="question">Le record de 35,8 MPa était-il prévisible par leur modèle ?</p>
          <p className="muted small" style={{ marginTop: 8 }}>Après 47 essais, le papier fige son XGBoost et laisse une optimisation bayésienne affiner r et T pour la résine 370 et le durcisseur 400. Les 4 propositions sont toutes prédites à 26,9 MPa.</p>
        </div>
        <div className="card controls">
          <h3>Recette</h3>
          <label>Résine époxy</label>
          <Seg options={{ 370: "370", 1650: "1 650", 2900: "2 900", 3800: "3 800" }} value={mwe} onChange={(v) => setMwe(+v)} />
          <label>Durcisseur</label>
          <Seg options={{ 230: "230", 400: "400", 2000: "2 000", 4000: "4 000" }} value={mwc} onChange={(v) => setMwc(+v)} />
          <label>Carte de droite</label>
          <Seg options={RIGHT} value={right} onChange={setRight} />
        </div>
        {d && (
          <div className="card">
            <h3>Ce qu’on observe</h3>
            <Stat num={d.plateaus.length}>valeurs différentes prédites par l’XGBoost sur tout le plan (r, T)</Stat>
            <Stat num={top.n.toLocaleString("fr-FR")}>points de la grille à égalité en tête ({fmt(top.value)} MPa)</Stat>
            {rec && <Stat num={pct(rec.p_beat)}>de chances, selon le GP, que la recette record batte 31,9 MPa</Stat>}
          </div>
        )}
        <div className="verdict">
          <h3>Lecture</h3>
          <p>Un modèle à arbres est constant par morceaux : affiner r au centième est invisible pour lui. Le GP donne une surface continue et une incertitude. Le record est un vrai résultat expérimental, pas une prédiction : il faut le répliquer.</p>
        </div>
      </aside>
      <section className="main">
        {!d ? <Loading error={err} /> : (
          <>
            <div className="grid2">
              <div className="panel">
                <h2>XGBoost figé (47 essais)</h2>
                <p className="sub">Des rectangles : quelques valeurs pour tout le plan.</p>
                <Plot config={cfg} useResizeHandler style={{ width: "100%", height: 400 }}
                  data={[heat(d.xgb, "MPa", "Viridis", 0, 32, ".1f"), boTrace]}
                  layout={baseLayout({ xaxis: { title: "ratio r" }, yaxis: { title: "cuisson (°C)" } })} />
              </div>
              <div className="panel">
                <h2>{RIGHT[right]} · GP sur les mêmes 47 essais</h2>
                <p className="sub">{right === "p_beat" ? "Probabilité que la vraie résistance dépasse 31,9 MPa." : right === "gp_sd" ? "Écart-type de la prédiction, en MPa." : "Prédiction continue, en MPa."}</p>
                <Plot config={cfg} useResizeHandler style={{ width: "100%", height: 400 }}
                  data={[right === "p_beat" ? heat(d.p_beat, "P(> 31,9)", "Magma", 0, 0.5, ".0%")
                    : right === "gp_sd" ? heat(d.gp_sd, "MPa", "Blues", 0, 8, ".1f")
                      : heat(d.gp_mean, "MPa", "Viridis", 0, 32, ".1f"), boTrace]}
                  layout={baseLayout({ xaxis: { title: "ratio r" }, yaxis: { title: "cuisson (°C)" } })} />
              </div>
            </div>
            {bo.length > 0 && (
              <div className="panel">
                <h2>Les 4 propositions de l’optimisation bayésienne</h2>
                <table>
                  <thead><tr><th>Essai</th><th>r</th><th>T (°C)</th><th>Mesuré</th><th>XGBoost</th><th>GP</th><th>Chance de battre 31,9</th></tr></thead>
                  <tbody>{bo.map((p) => (
                    <tr key={p.id}><td>n°{p.id}</td><td>{fmt(p.r, 2)}</td><td>{p.T}</td><td><b>{fmt(p.measured)}</b></td><td>{fmt(p.xgb)}</td><td>{fmt(p.gp_mean)} ± {fmt(p.gp_sd)}</td><td>{pct(p.p_beat)}</td></tr>
                  ))}</tbody>
                </table>
                <p className="sub" style={{ marginTop: 8 }}>Les n°49 et n°50 sont presque la même recette, et pourtant 27,4 contre 18,8 MPa : le bruit réel dépasse les ± 1,3 MPa annoncés.</p>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ 4. Simulator */
function Simulator() {
  const [oracle, setOracle] = useState("gp32");
  const [strategy, setStrategy] = useState("greedy_xgb");
  const [nInit, setNInit] = useState(10);
  const [seed, setSeed] = useState(0);
  const [runs, setRuns] = useState([]);
  const [busy, setBusy] = useState(false);
  const [bench] = useApi("/api/benchmark");

  const run = async (strategies) => {
    setBusy(true);
    const out = [];
    for (const s of strategies) {
      const r = await fetch(`/api/simulate?oracle=${oracle}&strategy=${s}&n_init=${nInit}&seed=${seed}`).then((x) => x.json());
      out.push({ ...r, strategy: s, oracle, nInit, seed });
      setRuns((prev) => [{ ...r, strategy: s, oracle, nInit, seed }, ...prev.filter((p) => !(p.strategy === s && p.oracle === oracle && p.nInit === nInit && p.seed === seed))].slice(0, 6));
    }
    setBusy(false);
  };
  useEffect(() => { setRuns([]); }, [oracle, nInit, seed]);
  const last = runs.find((r) => r.strategy === strategy) || runs[0];
  const benchRows = useMemo(() => (bench ? bench.rows.filter((r) => r.oracle === oracle && r.n_init === nInit) : []), [bench, oracle, nInit]);

  return (
    <div className="page">
      <aside className="side">
        <div className="card">
          <h3>La question</h3>
          <p className="question">Une autre stratégie aurait-elle mieux dépensé les 15 essais ?</p>
          <p className="muted small" style={{ marginTop: 8 }}>On ne peut pas refaire la campagne au labo. On fabrique donc un <b>faux labo</b> : un modèle entraîné sur les vraies mesures donne la « vraie » résistance de chaque recette, plus 2 MPa de bruit à chaque essai.</p>
        </div>
        <div className="card controls">
          <h3>Réglages</h3>
          <label>Faux labo construit avec…</label>
          <Seg options={ORACLE} value={oracle} onChange={setOracle} />
          <label>Départ</label>
          <Seg options={{ 32: "32 essais (le plan du papier)", 10: "10 essais" }} value={nInit} onChange={(v) => setNInit(+v)} />
          <label>Stratégie</label>
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            {Object.entries(STRAT).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <label>Tirage aléatoire (seed)
            <input type="number" min={0} max={99} value={seed} onChange={(e) => setSeed(+e.target.value)} />
          </label>
          <button className="go" disabled={busy} onClick={() => run([strategy])}>{busy ? "Campagne en cours…" : "Lancer la campagne"}</button>
          <button className="go" style={{ background: "#14213D" }} disabled={busy} onClick={() => run(Object.keys(STRAT))}>Comparer les 6 stratégies</button>
        </div>
        <div className="verdict">
          <h3>Lecture</h3>
          <p>Sur ce problème à 4 variables, avec un bon plan de départ, les stratégies se valent à 1 MPa près. Un GP sans incertitude fait aussi bien que la BO : ce qui aide, c’est un modèle lisse, pas l’exploration.</p>
          <p className="small" style={{ marginTop: 8, color: "#BFC8D8" }}>Limite : les faux labos héritent des 51 mesures. C’est une indication, pas une preuve.</p>
        </div>
      </aside>
      <section className="main">
        <div className="panel">
          <h2>Meilleure vraie résistance trouvée</h2>
          <p className="sub">{last ? `Optimum de ce faux labo : ${fmt(last.optimum.true)} MPa. Le tableau montre la stratégie choisie dans la liste.` : "Lancez une campagne pour voir les essais tomber, lot par lot. Avec 10 essais au départ, les écarts se voient mieux."}</p>
          <Plot config={cfg} useResizeHandler style={{ width: "100%", height: 340 }}
            data={[
              ...runs.map((r) => ({ x: [0, 5, 10, 15], y: r.best_so_far, name: STRAT[r.strategy], mode: "lines+markers", line: { color: STRAT_COLOR[r.strategy], width: 3 }, marker: { size: 8 } })),
              ...(last ? [{ x: [0, 15], y: [last.optimum.true, last.optimum.true], mode: "lines", name: "optimum du faux labo", line: { color: C.navy, dash: "dot" } }] : []),
            ]}
            layout={baseLayout({ xaxis: { title: "essais dépensés après le départ", tickvals: [0, 5, 10, 15] }, yaxis: { title: "MPa", gridcolor: "#eee" } })} />
        </div>
        {last && (
          <div className="panel">
            <h2>Les 15 essais choisis · {STRAT[last.strategy]}</h2>
            <table>
              <thead><tr><th>Lot</th><th>Résine</th><th>Durcisseur</th><th>r</th><th>T (°C)</th><th>Mesuré</th><th>Vrai</th></tr></thead>
              <tbody>{last.batches.flatMap((b, i) => b.map((p, j) => (
                <tr key={`${i}-${j}`}><td>{j === 0 ? `Lot ${i + 1}` : ""}</td><td>{p.MW_epoxy}</td><td>{p.MW_curing_agent}</td><td>{fmt(p.r, 2)}</td><td>{p.T_cure}</td><td>{fmt(p.measured)}</td><td><b>{fmt(p.true)}</b></td></tr>
              )))}</tbody>
            </table>
          </div>
        )}
        {benchRows.length > 0 && (
          <div className="panel">
            <h2>Sur 20 campagnes (calculé à l’avance)</h2>
            <p className="sub">Moyenne et intervalle de confiance à 95 % de la meilleure vraie résistance après 15 essais, pour ce faux labo et ce départ. Pointillé : la meilleure recette possible.</p>
            <Plot config={cfg} useResizeHandler style={{ width: "100%", height: 300 }}
              data={[{
                type: "scatter", mode: "markers", orientation: "h",
                y: Object.keys(STRAT).map((s) => STRAT[s]),
                x: Object.keys(STRAT).map((s) => benchRows.find((r) => r.strategy === s)?.mean),
                error_x: { type: "data", array: Object.keys(STRAT).map((s) => 1.96 * (benchRows.find((r) => r.strategy === s)?.sem || 0)), color: C.ink2 },
                marker: { size: 12, color: Object.keys(STRAT).map((s) => STRAT_COLOR[s]) }, hovertemplate: "%{y} : %{x:.1f} MPa<extra></extra>",
              }]}
              layout={baseLayout({ margin: { l: 200, r: 16, t: 24, b: 40 }, showlegend: false,
                xaxis: { title: "MPa", gridcolor: "#eee", range: [Math.min(...benchRows.map((r) => r.mean)) - 1, bench.optimum[oracle] + 0.5] },
                shapes: [{ type: "line", x0: bench.optimum[oracle], x1: bench.optimum[oracle], yref: "paper", y0: 0, y1: 1, line: { dash: "dot", color: C.navy } }],
                annotations: [{ x: bench.optimum[oracle], yref: "paper", y: 1.02, yanchor: "bottom", text: "optimum", showarrow: false, font: { size: 12 } }] })} />
          </div>
        )}
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ code + shell */
function Code({ tab }) {
  const [d] = useApi(`/api/code/${tab}`);
  return (
    <details className="code">
      <summary>Voir le code de cet onglet</summary>
      {d?.map((f) => (
        <div key={f.name}>
          <p className="fname">{f.file} · {f.name}</p>
          <pre>{f.source}</pre>
        </div>
      ))}
    </details>
  );
}

const TABS = [
  { id: "campaign", label: "La campagne", el: Campaign },
  { id: "loocv", label: "Le R²", el: R2 },
  { id: "replay", label: "Le record", el: Replay },
  { id: "simulate", label: "Le simulateur", el: Simulator },
];

export default function App() {
  const initial = TABS.find((t) => t.id === window.location.hash.slice(1))?.id || "campaign";
  const [tab, setTab] = useState(initial);
  useEffect(() => { window.location.hash = tab; }, [tab]);
  useEffect(() => {
    const on = () => { const h = window.location.hash.slice(1); if (TABS.some((t) => t.id === h)) setTab(h); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const T = TABS.find((t) => t.id === tab);
  return (
    <>
      <header className="top">
        <p className="kicker">Lecture critique · Pruksawan et al., 2019 · adhésif époxy, 51 essais</p>
        <h1>Que peut-on vraiment attribuer à la méthode ?</h1>
        <nav className="tabs">
          {TABS.map((t, i) => (
            <button key={t.id} className={`tab ${tab === t.id ? "on" : ""}`} onClick={() => setTab(t.id)}><b>{i + 1}</b>{t.label}</button>
          ))}
        </nav>
      </header>
      <T.el key={tab} />
      <Code tab={tab} />
    </>
  );
}
