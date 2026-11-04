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

// Small "?" badge with an explanation shown on hover or keyboard focus
function Help({ children }) {
  return (
    <span className="help" tabIndex={0} aria-label="Aide">
      ?
      <span className="help-pop" role="tooltip">{children}</span>
    </span>
  );
}

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
  const RIGHT = { p_beat: "Chance de battre 31,9", gp_mean: "Prédiction du GP" };
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
                <p className="sub">{right === "p_beat" ? "Probabilité que la vraie résistance dépasse 31,9 MPa." : "Prédiction continue, en MPa."}</p>
                <Plot config={cfg} useResizeHandler style={{ width: "100%", height: 400 }}
                  data={[right === "p_beat" ? heat(d.p_beat, "P(> 31,9)", "Magma", 0, 0.5, ".0%")
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
const STRAT_SHORT = { random: "Hasard", greedy_xgb: "Papier", greedy_rf: "Top 5 RF", gp_mean: "GP moyenne", gp_ei: "GP + EI", gp_ucb: "GP + UCB" };

function Takeaways({ scenarios, oracle, nInit, onPick }) {
  const rows = ["gp32", "gp51", "rf51", "xgb51"].flatMap((o) => [32, 10].map((n) => scenarios.find((s) => s.oracle === o && s.n_init === n))).filter(Boolean);
  return (
    <div className="panel takeaways">
      <h2>Ce qu’il faut retenir de la simulation</h2>
      <p className="sub">Résultats moyens sur 20 campagnes, pour les 4 faux labos et les 2 départs, soit 8 scénarios × 6 stratégies × 20 tirages.</p>
      <div className="keys">
        <div className="key"><span className="kn">1</span><div><b>La règle de choix change peu de chose.</b> Dans 5 scénarios sur 8, toutes les stratégies à modèle finissent à moins de 1 MPa les unes des autres. La méthode du papier n’est battue de façon solide que sur les deux faux labos GP, et de 0,06 à 0,56 MPa seulement.</div></div>
        <div className="key"><span className="kn">2</span><div><b>Un modèle ne bat le hasard que si le paysage est lisse.</b> Sur le faux labo « GP · 32 essais », toutes les stratégies à modèle approchent l’optimum et le hasard reste à 1,9 MPa. Sur les trois autres, le hasard fait aussi bien que le papier, aux erreurs près.</div></div>
        <div className="key"><span className="kn">3</span><div><b>Explorer n’apporte rien de mesurable.</b> GP + EI ne fait jamais mieux que le GP qui vise simplement la meilleure prédiction (aucun écart significatif sur les 8 scénarios). Le « + EI » n’est pas ce qui fait la différence.</div></div>
      </div>
      <div className="knobs">
        <p><b>Ce que change le faux labo.</b> Il fixe la difficulté, pas la conclusion. Un GP donne un paysage lisse, où une bonne recette annonce ses voisines. Une forêt aléatoire ou XGBoost font de la mesure record de 35,8 MPa un pic isolé, presque impossible à retrouver : d’où des écarts de 3 à 6 MPa pour toutes les stratégies, hasard compris.</p>
        <p><b>Ce que change le départ.</b> Avec 32 essais, on est dans la situation exacte du papier : le plan couvre déjà bien l’espace et 15 essais de plus gagnent peu, parfois presque rien. Avec 10 essais, il reste plus à chercher, mais les stratégies à modèle restent proches entre elles.</p>
        <p className="bottom"><b>En une phrase :</b> sur ce problème, le bon plan d’expériences de départ fait l’essentiel. Avant d’attribuer un gain à l’active learning, il faut le comparer au hasard, et sur plusieurs faux labos.</p>
      </div>
      <div className="tscroll"><table className="scen">
        <thead><tr><th>Faux labo</th><th>Départ<br /><span className="small">essais</span></th><th>Avant les<br />15 essais</th><th>Hasard</th><th>Papier</th><th>Gagné en 15 essais<br /><span className="small">papier · hasard</span></th><th>Meilleure autre</th><th>Bat le papier ?</th></tr></thead>
        <tbody>{rows.map((r) => {
          const on = r.oracle === oracle && r.n_init === nInit;
          const d = r.gaps.greedy_xgb - r.gaps[r.best];
          const sig = r.p_best_vs_paper < 0.05;
          return (
            <tr key={`${r.oracle}-${r.n_init}`} className={on ? "on" : ""} onClick={() => onPick(r.oracle, r.n_init)} title="Afficher ce scénario">
              <td>{ORACLE[r.oracle].replace("Random Forest", "Forêt aléatoire")}</td><td>{r.n_init}</td><td>{fmt(r.start_gap)}</td>
              <td>{fmt(r.gaps.random)}</td><td className="paper">{fmt(r.gaps.greedy_xgb)}</td>
              <td><b>{fmt(r.start_gap - r.gaps.greedy_xgb)}</b> <span className="small">· {fmt(r.start_gap - r.gaps.random)}</span></td>
              <td>{fmt(r.gaps[r.best])} <span className="small">{STRAT_SHORT[r.best]}</span></td>
              <td className={sig ? "yes" : "no"}>{sig ? `oui, de ${fmt(d, 2)}` : "non"}</td>
            </tr>);
        })}</tbody>
      </table></div>
      <p className="small">MPa sous l’optimum, 0 = meilleure recette trouvée. « Gagné en 15 essais » : écart avant moins écart après, pour le papier puis pour le hasard. « Bat le papier ? » : la meilleure autre stratégie fait-elle mieux que le papier (test de Wilcoxon apparié sur les 20 tirages, seuil 5 %) ? Si oui, de combien de MPa. Cliquez une ligne pour l’afficher plus bas.</p>
      <p className="small">Limite : les faux labos sont construits à partir des 51 mesures. C’est une indication, pas une preuve.</p>
    </div>
  );
}

function scenText(r) {
  const g = r.gaps, d = g.greedy_xgb - g[r.best];
  const vsBest = r.p_best_vs_paper < 0.05
    ? `${STRAT_SHORT[r.best]} fait mieux que le papier, de ${fmt(d, 2)} MPa en moyenne.`
    : `Aucune stratégie ne bat le papier de façon solide : l’écart avec ${STRAT_SHORT[r.best]} (${fmt(d, 2)} MPa) reste dans le bruit.`;
  const vsRand = g.random - g.greedy_xgb > 0.5 && r.p_random_vs_paper < 0.05
    ? "Le hasard est nettement moins bon : ici, le modèle sert vraiment."
    : "Le hasard fait presque aussi bien : ici, le modèle n’apporte pas grand-chose.";
  const gain = r.start_gap - g.greedy_xgb;
  return `Les 15 essais du papier font gagner ${fmt(gain)} MPa. ${vsBest} ${vsRand}`;
}

function Simulator() {
  const [oracle, setOracle] = useState("gp32");
  const [strategy, setStrategy] = useState("greedy_xgb");
  const [nInit, setNInit] = useState(10);
  const [seed, setSeed] = useState(0);
  const [runs, setRuns] = useState({});
  const [busy, setBusy] = useState(false);
  const [bench] = useApi("/api/benchmark");

  // One campaign per strategy for the current settings, run as soon as they change
  useEffect(() => {
    let alive = true;
    setRuns({}); setBusy(true);
    (async () => {
      for (const s of Object.keys(STRAT)) {
        const r = await fetch(`/api/simulate?oracle=${oracle}&strategy=${s}&n_init=${nInit}&seed=${seed}`).then((x) => x.json());
        if (!alive) return;
        setRuns((prev) => ({ ...prev, [s]: r }));
      }
      if (alive) setBusy(false);
    })();
    return () => { alive = false; };
  }, [oracle, nInit, seed]);

  const any = Object.values(runs)[0];
  const opt = any?.optimum.true;
  const sel = runs[strategy];
  const gap = (r) => opt - r.best_so_far[r.best_so_far.length - 1];
  const benchRows = useMemo(() => (bench ? bench.rows.filter((r) => r.oracle === oracle && r.n_init === nInit) : []), [bench, oracle, nInit]);
  const benchGap = (s) => { const r = benchRows.find((x) => x.strategy === s); return r ? { gap: bench.optimum[oracle] - r.mean, ci: 1.96 * r.sem } : null; };
  const paperGap = benchGap("greedy_xgb"), randomGap = benchGap("random");
  const scen = bench?.scenarios?.find((x) => x.oracle === oracle && x.n_init === nInit);

  // Staircase lines: the selected strategy and the paper in colour, the others in light grey
  const lines = Object.entries(runs).map(([s, r]) => {
    const on = s === strategy || s === "greedy_xgb";
    return {
      x: [0, 5, 10, 15], y: r.best_so_far.map((b) => opt - b), name: STRAT[s], mode: "lines+markers",
      line: { shape: "hv", color: on ? STRAT_COLOR[s] : "#C9CFD9", width: on ? 4 : 2 },
      marker: { size: on ? 9 : 6, color: on ? STRAT_COLOR[s] : "#C9CFD9" },
      hovertemplate: `${STRAT[s]}<br>après %{x} essais : %{y:.1f} MPa sous l’optimum<extra></extra>`,
      showlegend: false,
    };
  }).sort((u, v) => (u.line.width - v.line.width));
  const labels = Object.entries(runs).map(([s, r]) => ({ s, y: opt - r.best_so_far[3] }))
    .sort((u, v) => u.y - v.y);
  // spread end labels so they do not overlap
  const ymax = Math.max(1, ...Object.values(runs).map((r) => opt - r.best_so_far[0]));
  const minGap = ymax * 0.07;
  labels.forEach((l, i) => { l.ly = i === 0 ? l.y : Math.max(l.y, labels[i - 1].ly + minGap); });

  return (
    <div className="page">
      <aside className="side">
        <div className="card">
          <h3>La question</h3>
          <p className="question">Avec les mêmes 15 essais, une autre façon de choisir les recettes aurait-elle trouvé une meilleure colle ?</p>
        </div>
        <div className="card">
          <h3>Comment ça marche</h3>
          <ol className="steps">
            <li><b>Un faux labo.</b> On ne peut pas refaire les essais. Un modèle entraîné sur les vraies mesures donne donc la « vraie » résistance de chacune des 3 328 recettes, et chaque essai ajoute 2 MPa de bruit.</li>
            <li><b>Le même départ pour tous.</b> Chaque stratégie part des mêmes essais, puis choisit 3 lots de 5 recettes, comme dans le papier.</li>
            <li><b>On compare</b> l’écart entre la meilleure recette trouvée et la meilleure recette possible : <b>0 = trouvée</b>.</li>
          </ol>
        </div>
        <div className="card controls">
          <h3>Réglages</h3>
          <label><span className="lab">Faux labo construit avec… <Help>Le modèle qui joue le rôle de la réalité : il donne la « vraie » résistance de chaque recette. « 51 mesures » : entraîné sur tous les essais du papier. « 32 essais initiaux » : sans l’active learning ni le record, pour que le faux labo ne connaisse pas déjà les bonnes réponses. Changer de faux labo vérifie que la conclusion ne dépend pas d’un seul modèle.</Help></span></label>
          <Seg options={ORACLE} value={oracle} onChange={setOracle} />
          <label><span className="lab">Départ <Help><b>32 essais</b> : tout le plan d’expériences du papier (deux carrés gréco-latins 4 × 4), c’est-à-dire leur situation exacte avant l’active learning. Le plan couvre déjà bien l’espace : il reste peu à gagner.<br /><br /><b>10 essais</b> : 10 de ces 32 essais, choisis pour être les plus éloignés les uns des autres. On en sait moins au départ, la stratégie doit vraiment chercher : c’est là que la règle de choix compte le plus.<br /><br />Dans les deux cas, chaque stratégie ajoute ensuite 15 essais, en 3 lots de 5.</Help></span></label>
          <Seg options={{ 32: "32 essais (le plan du papier)", 10: "10 essais" }} value={nInit} onChange={(v) => setNInit(+v)} />
          <label><span className="lab">Stratégie mise en avant <Help>Les 6 stratégies tournent toujours. Celle-ci s’affiche en couleur à côté de la méthode du papier (en orange), et c’est son détail qui apparaît dans le tableau des 15 essais.</Help></span></label>
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            {Object.entries(STRAT).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <label><span className="lab">Tirage aléatoire (seed) <Help>Fixe le hasard d’une campagne : le bruit de mesure (± 2 MPa par essai), le tirage au sort entre recettes à égalité et, avec 10 essais, lesquels des 32 servent de départ. Même numéro = même résultat. Une campagne seule peut avoir de la chance : c’est la moyenne sur 20 tirages, dans le tableau en haut et la carte « Ce scénario », qui compte.</Help></span>
            <input type="number" min={0} max={99} value={seed} onChange={(e) => setSeed(Math.max(0, Math.min(99, +e.target.value)))} />
          </label>
          <button className="go" disabled={busy} onClick={() => setSeed((seed + 1) % 100)}>{busy ? "Campagnes en cours…" : "Rejouer avec un autre tirage"}</button>
        </div>
        {paperGap && randomGap && scen && (
          <div className="card">
            <h3>Ce scénario, sur 20 campagnes <Help>Les 3 lots de 5 essais sont rejoués 20 fois, avec 20 tirages différents du bruit de mesure (et, avec 10 essais, du départ). Une seule campagne peut avoir de la chance ou pas : la moyenne de 20 dit ce qu’on peut attendre d’une stratégie. Tous les chiffres sont en MPa sous la meilleure recette possible, 0 = trouvée.</Help></h3>
            <div className="stat"><span className="num">{fmt(scen.start_gap)}</span><span>avant les 15 essais <Help>L’écart entre la meilleure recette du départ et la meilleure recette possible. C’est le point de départ commun à toutes les stratégies.</Help></span></div>
            <div className="stat"><span className="num">{fmt(paperGap.gap)}</span><span>après 15 essais, avec la méthode du papier <Help>La règle du papier : à chaque lot, tester les 5 recettes que XGBoost prédit les meilleures. La différence avec la ligne du dessus, c’est ce que les 15 essais ont fait gagner.</Help></span></div>
            <div className="stat"><span className="num">{fmt(scen.gaps[scen.best])}</span><span>avec la meilleure autre stratégie ({STRAT_SHORT[scen.best]}) <Help>La meilleure des 4 autres stratégies à modèle sur ce scénario. Si l’écart avec le papier est petit, ou pas significatif, changer de règle de choix n’aurait rien changé.</Help></span></div>
            <div className="stat"><span className="num">{fmt(randomGap.gap)}</span><span>en choisissant au hasard <Help>15 recettes tirées au sort, sans modèle. C’est la référence : si une stratégie à modèle ne fait pas nettement mieux, le modèle n’apporte rien ici.</Help></span></div>
            <p className="small" style={{ marginTop: 8 }}>{scenText(scen)}</p>
          </div>
        )}
      </aside>
      <section className="main">
        {bench?.scenarios && <Takeaways scenarios={bench.scenarios} oracle={oracle} nInit={nInit} onPick={(o, n) => { setOracle(o); setNInit(n); }} />}
        <div className="panel">
          <h2>Une campagne en exemple : combien manque-t-il pour atteindre la meilleure recette ?</h2>
          <p className="sub">Une seule campagne de 15 essais par stratégie, même départ et même tirage. Cliquez « Rejouer avec un autre tirage » : le classement change d’une campagne à l’autre, d’où les moyennes sur 20 campagnes plus haut. Plus la marche descend, plus la stratégie s’approche de l’optimum ({opt ? `${fmt(opt)} MPa` : "…"}). En couleur : la méthode du papier et la stratégie choisie.</p>
          {!any ? <Loading /> : (
            <Plot config={cfg} useResizeHandler style={{ width: "100%", height: 360 }}
              data={lines}
              layout={baseLayout({
                margin: { l: 64, r: 130, t: 16, b: 52 }, showlegend: false,
                xaxis: { title: "essais dépensés après le départ", tickvals: [0, 5, 10, 15], ticktext: ["départ", "lot 1", "lot 2", "lot 3"], range: [-0.4, 15.4], zeroline: false },
                yaxis: { title: "MPa sous l’optimum", range: [-0.06 * Math.max(ymax, labels[labels.length - 1]?.ly || 0), 1.08 * Math.max(ymax, labels[labels.length - 1]?.ly || 0)], gridcolor: "#eee", zeroline: true, zerolinecolor: C.navy, zerolinewidth: 1.5 },
                annotations: [
                  ...labels.map((l) => ({ x: 15, y: l.ly, xanchor: "left", xshift: 10, showarrow: false, text: STRAT_SHORT[l.s],
                    font: { size: 12, color: l.s === strategy || l.s === "greedy_xgb" ? STRAT_COLOR[l.s] : "#9AA3B2" } })),
                  { x: 0, y: 0, xanchor: "left", yanchor: "bottom", showarrow: false, text: "0 = meilleure recette trouvée", font: { size: 11, color: C.ink2 } },
                ],
              })} />
          )}
        </div>
        {sel && (
          <details className="panel fold">
            <summary><h2>Voir les 15 essais · {STRAT[strategy]}</h2></summary>
            <p className="sub">« Mesuré » : ce que le faux labo renvoie, bruit compris. « Vrai » : la résistance réelle de la recette.</p>
            <table>
              <thead><tr><th>Lot</th><th>Résine</th><th>Durcisseur</th><th>r</th><th>T (°C)</th><th>Mesuré</th><th>Vrai</th></tr></thead>
              <tbody>{sel.batches.flatMap((b, i) => b.map((p, j) => (
                <tr key={`${i}-${j}`}><td>{j === 0 ? `Lot ${i + 1}` : ""}</td><td>{p.MW_epoxy}</td><td>{p.MW_curing_agent}</td><td>{fmt(p.r, 2)}</td><td>{p.T_cure}</td><td>{fmt(p.measured)}</td><td><b>{fmt(p.true)}</b></td></tr>
              )))}</tbody>
            </table>
          </details>
        )}
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ shell */
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
    </>
  );
}
