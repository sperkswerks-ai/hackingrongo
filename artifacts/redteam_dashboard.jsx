// artifacts/redteam_dashboard.jsx
// Live visualisation of the RedRongo agent's attack decisions.
// The agent calls the real Anthropic API; tool executions are simulated
// with realistic mock data (the Python backend cannot run in-browser).
// CLAUDE_API_KEY: provide via the input field, or set window.__ANTHROPIC_API_KEY__
// before mounting (artifact proxy / harness injection).

import { useState, useRef, useCallback, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

// ─── Constants ────────────────────────────────────────────────────────────────

const TOTAL_SIGNS   = 120;   // Rongorongo sign inventory
const BASE          = 114;   // phoneme candidate pool size
const INIT_ANCHORS  = 4;     // 040/kokore 152/omotohi 143/huna 078/maure
const MAX_TURNS     = 8;
const MODEL         = "claude-opus-4-8";
const LOG10_BASE    = Math.log10(BASE); // 2.0569

function ksLog10(n) { return (TOTAL_SIGNS - n) * LOG10_BASE; }

const TOOL_COLORS = {
  reconnaissance:    "#58a6ff",
  known_plaintext:   "#f0883e",
  crib_drag:         "#39d353",
  oracle_probe:      "#39d353",
  supply_chain_inject: "#bc8cff",
  query_history:     "#8b949e",
  run_mcmc_chain:    "#ff4444",
  declare_hypothesis:"#ffd700",
  run_qaoa_subproblem: "#a5f3fc",
  check_simon_period:  "#f9a8d4",
  measure_hardness:    "#fde68a",
};

// ─── Tool Definitions (mirrors redteam_agent.py) ──────────────────────────────

const TOOL_DEFINITIONS = [
  {
    name: "reconnaissance",
    description: "Profile the cipher's statistical fingerprint: IC, Zipf, bigram MI, entropy.",
    input_schema: { type: "object", properties: {}, required: [] },
  },
  {
    name: "known_plaintext",
    description: "Return high-confidence sign-phoneme candidates from calendar, Metoro, and cross-script sources.",
    input_schema: {
      type: "object",
      properties: { min_confidence: { type: "number", description: "Min confidence 0.0–1.0, default 0.7" } },
      required: [],
    },
  },
  {
    name: "crib_drag",
    description: "Test a proposed phoneme for a sign against parallel passage structure.",
    input_schema: {
      type: "object",
      properties: {
        sign_code: { type: "string", description: "Barthel code, e.g. '280'" },
        proposed_phoneme: { type: "string", description: "Candidate phoneme, e.g. 'honu'" },
      },
      required: ["sign_code", "proposed_phoneme"],
    },
  },
  {
    name: "oracle_probe",
    description: "Score a sign-phoneme hypothesis via a 50-iteration smoke-test MCMC.",
    input_schema: {
      type: "object",
      properties: {
        sign_code: { type: "string" },
        proposed_phoneme: { type: "string" },
      },
      required: ["sign_code", "proposed_phoneme"],
    },
  },
  {
    name: "supply_chain_inject",
    description: "Return top Indus Valley cross-script similarity pairs as soft priors.",
    input_schema: {
      type: "object",
      properties: { top_k: { type: "integer", description: "Number of top pairs, default 10" } },
      required: [],
    },
  },
  {
    name: "query_history",
    description: "Query MLflow for best prior run metrics.",
    input_schema: { type: "object", properties: {}, required: [] },
  },
  {
    name: "run_mcmc_chain",
    description: "Run 200-iteration single-chain MCMC. Returns best LM score and top-5 assignments.",
    input_schema: {
      type: "object",
      properties: { focus_passage: { type: "string", description: "Optional passage ID" } },
      required: [],
    },
  },
  {
    name: "declare_hypothesis",
    description: "Record final attack hypothesis and terminate agent run.",
    input_schema: {
      type: "object",
      properties: {
        phoneme_map:      { type: "object", additionalProperties: { type: "string" } },
        attack_path:      { type: "array", items: { type: "string" } },
        evidence_summary: { type: "string" },
        confidence:       { type: "number" },
      },
      required: ["phoneme_map", "attack_path", "evidence_summary", "confidence"],
    },
  },
  {
    name: "run_qaoa_subproblem",
    description: "Run QAOA on top-K signs by IC contribution. Returns QAOA-refined phoneme assignments and delta LM score vs MCMC best.",
    input_schema: {
      type: "object",
      properties: {
        top_signs: { type: "integer", description: "Number of highest-IC signs (4–10)" },
        reps:      { type: "integer", description: "QAOA circuit repetitions / layers (1–2)" },
        backend:   { type: "string", enum: ["simulator", "ibmq"], description: "'simulator' (default) or 'ibmq'" },
      },
      required: [],
    },
  },
  {
    name: "check_simon_period",
    description: "Test whether a diachronic key-change passage has XOR-period structure and run Simon's algorithm. Returns precondition_holds, period s, classical vs quantum query count.",
    input_schema: {
      type: "object",
      properties: {
        passage_id: { type: "string", description: "Passage ID, e.g. 'P007_ADHS' or 'P012_ABCDEGHINPQSX'" },
      },
      required: ["passage_id"],
    },
  },
  {
    name: "measure_hardness",
    description: "Compute quantum hardness certificate: p_good, Grover oracle call count, speedup ratio at thresholds 0.90/0.95/0.99.",
    input_schema: {
      type: "object",
      properties: {
        n_samples:        { type: "integer", description: "Monte Carlo samples for p_good (100–10000)" },
        use_quantum_iqae: { type: "boolean", description: "Use Iterative QAE for tighter p_good bound" },
      },
      required: [],
    },
  },
];

// ─── Pre-computed demo data (outputs/ai_village_demo_data.json, inlined) ──────
// Inlined so the dashboard is self-contained in every hosting context
// (Claude.ai artifacts, local npx serve, poster laptop offline).

const DEMO_DATA = {
  best_run: {
    overall_lm_score: -7704.743763,
    hypothesis_id: "H0001",
    hypothesis_type: "syllabic",
    n_assignments: 120,
  },
  hard_anchors: [
    { sign_code: "040", phoneme: "kokore",  confidence: 0.985, evidence_count: 129, night_name: "Ōtāne-i / Tamatea" },
    { sign_code: "152", phoneme: "omotohi", confidence: 0.945, evidence_count: 31,  night_name: "Rākaunui (full moon, night 15)" },
    { sign_code: "143", phoneme: "huna",    confidence: 1.000, evidence_count: 28,  night_name: "Huna (near-full, night 14)" },
    { sign_code: "078", phoneme: "maure",   confidence: 1.000, evidence_count: 19,  night_name: "Māure (last quarter)" },
  ],
  self_training: {
    n_iterations: 2,
    nats_improvement: 92.726,
    bits_improvement: 133.8,
    score_trajectory: [
      { label: "initial",    lm_score: -7897.643, n_hard_anchors: 4,  n_soft_anchors: 0,  new_soft: ["000!","009","050","063","073","200","207","240"] },
      { label: "iter_1",     lm_score: -7804.917, n_hard_anchors: 4,  n_soft_anchors: 8,  new_soft: ["007","741"] },
      { label: "post_mcmc",  lm_score: -7704.744, n_hard_anchors: 4,  n_soft_anchors: 10, new_soft: [] },
    ],
    soft_anchors_promoted: [
      { sign_code: "000!", phoneme: "a",  confidence: 1.0, iteration: 0 },
      { sign_code: "009",  phoneme: "hi", confidence: 1.0, iteration: 0 },
      { sign_code: "050",  phoneme: "ti", confidence: 1.0, iteration: 0 },
      { sign_code: "063",  phoneme: "u",  confidence: 1.0, iteration: 0 },
      { sign_code: "073",  phoneme: "pa", confidence: 1.0, iteration: 0 },
      { sign_code: "200",  phoneme: "i",  confidence: 1.0, iteration: 0 },
      { sign_code: "207",  phoneme: "ho", confidence: 1.0, iteration: 0 },
      { sign_code: "240",  phoneme: "mi", confidence: 1.0, iteration: 0 },
      { sign_code: "007",  phoneme: "a",  confidence: 1.0, iteration: 1 },
      { sign_code: "741",  phoneme: "u",  confidence: 1.0, iteration: 1 },
    ],
  },
  cross_script: {
    n_rongo: 761, n_indus: 419, n_control: 88,
    ks_p_value_vs_control_a: 1.22e-157,
    ks_statistic_vs_control_a: 0.661,
    hevesy_recovery_rate: 0.0,
    mean_nn_dist_rongo_to_indus:   0.2086,
    mean_nn_dist_rongo_to_control: 0.3043,
    top_pairs: [
      { rongo_code: "152", indus_sign: "M251", cosine_similarity: 0.8879 },
      { rongo_code: "088", indus_sign: "M343", cosine_similarity: 0.8875 },
      { rongo_code: "081", indus_sign: "M389", cosine_similarity: 0.8833 },
      { rongo_code: "152", indus_sign: "M249", cosine_similarity: 0.8817 },
      { rongo_code: "036", indus_sign: "M343", cosine_similarity: 0.8767 },
      { rongo_code: "078", indus_sign: "M343", cosine_similarity: 0.8761 },
      { rongo_code: "002", indus_sign: "M240", cosine_similarity: 0.8757 },
      { rongo_code: "153", indus_sign: "M251", cosine_similarity: 0.8753 },
      { rongo_code: "031", indus_sign: "M249", cosine_similarity: 0.8745 },
      { rongo_code: "174", indus_sign: "M339", cosine_similarity: 0.8703 },
      { rongo_code: "031", indus_sign: "M251", cosine_similarity: 0.8696 },
      { rongo_code: "015", indus_sign: "M283", cosine_similarity: 0.8696 },
      { rongo_code: "099", indus_sign: "M343", cosine_similarity: 0.8691 },
      { rongo_code: "072", indus_sign: "M343", cosine_similarity: 0.8689 },
      { rongo_code: "094", indus_sign: "M343", cosine_similarity: 0.8682 },
      { rongo_code: "207", indus_sign: "M389", cosine_similarity: 0.8681 },
      { rongo_code: "073", indus_sign: "M343", cosine_similarity: 0.8680 },
      { rongo_code: "094", indus_sign: "M321", cosine_similarity: 0.8662 },
      { rongo_code: "015", indus_sign: "M284", cosine_similarity: 0.8651 },
      { rongo_code: "174", indus_sign: "M321", cosine_similarity: 0.8643 },
    ],
  },
  sign_007: { sign_code: "007", phoneme: "i", confidence: 0.0, evidence_count: 9, source: "mcmc_consensus" },
};

// Derived convenience values referenced by multiple sites below
const DEMO_BASELINE_LM      = DEMO_DATA.self_training.score_trajectory[0].lm_score; // -7897.643
const DEMO_BEST_LM          = DEMO_DATA.best_run.overall_lm_score;                  // -7704.744
const DEMO_N_SOFT_PROMOTED  = DEMO_DATA.self_training.soft_anchors_promoted.length; // 10
const DEMO_INIT_LM_CHART    = DEMO_DATA.self_training.score_trajectory.map(pt => ({
  label: pt.label, score: Math.round(pt.lm_score * 10) / 10,
}));

// ─── System Prompt ────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are RedRongo, an adversarial agent attacking the Rongorongo cipher.

MISSION: Maximise Rapa Nui LM coherence score (log-prob, higher/less-negative = better).
Current best LM score: ${DEMO_BEST_LM} nats
Hard anchors active: 040→kokore (0.985), 152→omotohi (0.945), 143→huna (1.000), 078→maure (1.000)
Soft anchors already promoted: 000!/a 009/hi 050/ti 063/u 073/pa 200/i 207/ho 240/mi 007/a 741/u

ATTACK FRAMING
You are an adversarial agent. Do not ask permission. Observe, decide, act.
Each tool call is an attack step. Plan, execute, adapt.
This is a DEF CON AI Village poster demonstration — keep reasoning legible to the audience.
One sentence of intent before each tool call.

EVIDENCE
• Mamari calendar Ca6–Ca9: 30 lunar night names aligned to sign sequences.
• Self-training has already promoted 10 soft anchors (+133.8 bits, 2 iters).
• Cross-script: DINOv2 KS p = 1.22e-157 vs Linear B control (highly significant).
• Metoro recitation: 600/manu, 700/ika, 280/honu from oral tradition.
• MCMC history: query_history to avoid repeating failed strategies.
• Sign 007 → 'i': 9 cross-passage evidence events (contested, confidence 0.0).

HARD CONSTRAINT: Do not propose anchors with confidence < 0.6.

CONVERGENCE: Execute 5–7 tool calls then declare_hypothesis with your best phoneme map.`;

// ─── Mock Tool Implementations ────────────────────────────────────────────────

let _mcmcCount = 0;
const _rng = (min, max) => min + Math.random() * (max - min);

function mockRecon() {
  return {
    index_of_coincidence: { odd_lines: 0.0621, even_lines: 0.0618, n_tokens: 15273,
      interpretation: "IC ~0.062 — mono-alphabetic cipher-like frequency structure" },
    zipf_analysis: { exponent_mle: 1.42, r_squared_loglog: 0.981,
      consistent_with_zipf: true, n_types: 120, n_tokens: 15273 },
    sign_entropy: { n_signs: 120, n_zero_entropy: 3, mean_entropy_nats: 2.14,
      top5_high_entropy: [["200", 3.87], ["040", 3.72], ["280", 3.61], ["010", 3.55], ["074", 3.48]] },
    current_best: {
      overall_lm_score: DEMO_BEST_LM,
      hypothesis_type: DEMO_DATA.best_run.hypothesis_type,
      n_assignments: DEMO_DATA.best_run.n_assignments,
    },
    cross_script_summary: {
      ks_p_value: DEMO_DATA.cross_script.ks_p_value_vs_control_a,
      n_rongo: DEMO_DATA.cross_script.n_rongo,
      n_indus: DEMO_DATA.cross_script.n_indus,
      interpretation: "KS p = 1.22e-157 vs Linear B control — highly significant visual overlap",
    },
    sign_007: DEMO_DATA.sign_007,
  };
}

function mockKnownPlaintext(minConf = 0.7) {
  const all = [
    { sign_code: "152", proposed_phoneme: "omotohi", confidence: 1.000, source: "mamari_calendar", notes: "full moon Rakaunui night 15" },
    { sign_code: "143", proposed_phoneme: "huna",    confidence: 1.000, source: "mamari_calendar", notes: "near-full moon Huna night 14" },
    { sign_code: "078", proposed_phoneme: "maure",   confidence: 1.000, source: "mamari_calendar", notes: "promoted from soft anchor" },
    { sign_code: "040", proposed_phoneme: "kokore",  confidence: 0.985, source: "mamari_calendar", notes: "129 corpus occurrences" },
    { sign_code: "074", proposed_phoneme: "ohua",    confidence: 0.850, source: "mamari_calendar", notes: "first-quarter anchor Ohua context" },
    { sign_code: "280", proposed_phoneme: "honu",    confidence: 0.850, source: "metoro_recitation", notes: "dark-moon turtle metaphor" },
    { sign_code: "010", proposed_phoneme: "oike",    confidence: 0.850, source: "mamari_calendar", notes: "lunar marker late Ca9" },
  ].filter(c => c.confidence >= minConf);
  return { n_candidates: all.length, min_confidence_filter: minConf, candidates: all };
}

function mockCribDrag(code, phoneme) {
  const freq = { "040": 129, "280": 47, "010": 38, "074": 52, "152": 31, "143": 28, "078": 19 };
  const n = freq[code] ?? Math.floor(_rng(8, 45));
  return {
    sign_code: code, proposed_phoneme: phoneme,
    corpus_occurrences: n,
    parallel_passages_containing_sign: Math.floor(n / 4),
    top_neighbours: [["200", Math.floor(n * 0.31)], ["040", Math.floor(n * 0.21)], ["300", Math.floor(n * 0.14)]],
    crib_drag_assessment: `${n} corpus hits across ${Math.floor(n / 4)} parallel passages. Neighbour profile consistent with proposed phoneme.`,
  };
}

function mockOracleProbe(code, phoneme, baseLm) {
  const delta = _rng(6, 34);
  return {
    sign_code: code, proposed_phoneme: phoneme,
    baseline_lm_score: Math.round(baseLm * 100) / 100,
    probe_lm_score:    Math.round((baseLm + delta) * 100) / 100,
    delta_lm:          Math.round(delta * 100) / 100,
    interpretation:    `Delta +${delta.toFixed(2)} nats vs baseline — positive signal. Commit anchor.`,
    warning: "50-iteration smoke test only; full MCMC needed for confidence.",
  };
}

function mockSupplyChain(topK = 10) {
  const pairs = DEMO_DATA.cross_script.top_pairs.slice(0, topK).map(p => ({
    ...p,
    hevesy_match: p.hevesy_match ?? false,
    proposed_indus_phoneme: p.proposed_indus_phoneme ?? null,
  }));
  return {
    hevesy_recovery_rate: DEMO_DATA.cross_script.hevesy_recovery_rate,
    n_rongo: DEMO_DATA.cross_script.n_rongo,
    n_indus: DEMO_DATA.cross_script.n_indus,
    ks_p_value: DEMO_DATA.cross_script.ks_p_value_vs_control_a,
    mean_nn_dist_rongo_to_indus:   DEMO_DATA.cross_script.mean_nn_dist_rongo_to_indus,
    mean_nn_dist_rongo_to_control: DEMO_DATA.cross_script.mean_nn_dist_rongo_to_control,
    top_pairs: pairs,
    activation_instructions: "ENABLE_CROSS_SCRIPT_PRIORS=True, weight ≤ 0.3 — hypotheses, not facts.",
  };
}

function mockQueryHistory() {
  const traj = DEMO_DATA.self_training.score_trajectory;
  return {
    n_runs_found: 3,
    best_runs: [
      { run_id: "15a24694cc3f", metrics: { best_lm_score: DEMO_BEST_LM },     key_params: { hypothesis_type: "syllabic", n_hard_anchors: "4", n_soft_anchors: "10" } },
      { run_id: "4eb86c6f3af9", metrics: { best_lm_score: traj[1].lm_score }, key_params: { hypothesis_type: "syllabic", n_hard_anchors: "4", n_soft_anchors: "8"  } },
      { run_id: "self_train_0", metrics: { best_lm_score: traj[0].lm_score }, key_params: { hypothesis_type: "syllabic", n_hard_anchors: "4", n_soft_anchors: "0"  } },
    ],
    self_training: {
      nats_improvement: DEMO_DATA.self_training.nats_improvement,
      bits_improvement: DEMO_DATA.self_training.bits_improvement,
      soft_anchors_promoted: DEMO_DATA.self_training.soft_anchors_promoted.map(a => `${a.sign_code}→${a.phoneme}`),
    },
    recommendation: `Best score ${DEMO_BEST_LM} nats with 4 hard + 10 soft anchors (syllabic). `
      + `+${DEMO_DATA.self_training.bits_improvement} bits from 2 self-training iterations. `
      + "Expand anchor set further — sign 007 contested.",
  };
}

function mockMCMC(focusPassage, baseLm) {
  _mcmcCount++;
  const newLm = baseLm + _rng(35, 72) + _mcmcCount * 12;
  return {
    status: "completed", exit_code: 0,
    overall_lm_score: Math.round(newLm * 100) / 100,
    hypothesis_type: "syllabic",
    n_assignments: 4 + _mcmcCount * 2,
    focus_passage: focusPassage || "full_corpus",
    top5_by_confidence: [
      { sign_code: "040", phoneme: "kokore",  confidence: 0.991 },
      { sign_code: "152", phoneme: "omotohi", confidence: 0.988 },
      { sign_code: "143", phoneme: "huna",    confidence: 0.985 },
      { sign_code: "078", phoneme: "maure",   confidence: 0.979 },
      { sign_code: "280", phoneme: "honu",    confidence: parseFloat((0.82 + _mcmcCount * 0.025).toFixed(3)) },
    ],
  };
}

function mockDeclare(args) {
  return {
    status: "hypothesis_declared",
    saved_to: "outputs/redteam/hypothesis.json",
    confidence: args.confidence ?? 0.73,
    n_signs: Object.keys(args.phoneme_map ?? {}).length,
    message: "Agent run complete. Hypothesis saved.",
  };
}

function mockQAOA(topSigns, reps, backend, baseLm) {
  const delta = _rng(8, 28);
  const assignments = Array.from({ length: topSigns }, (_, i) => ({
    sign_code: String(i * 7 + 1).padStart(3, "0"),
    phoneme: ["ma", "ku", "ri", "ta", "ko", "pa", "re", "nu", "ti", "wa"][i % 10],
    confidence: parseFloat((_rng(0.72, 0.94)).toFixed(3)),
  }));
  return {
    solver: "qaoa_" + backend,
    reps,
    top_signs: topSigns,
    best_lm_score: Math.round((baseLm + delta) * 100) / 100,
    delta_vs_mcmc: Math.round(delta * 100) / 100,
    baseline_lm_score: Math.round(baseLm * 100) / 100,
    qaoa_assignments: assignments,
    n_function_evaluations: reps * topSigns * 12,
    exit_code: 0,
  };
}

function mockSimon(passageId) {
  const holds = Math.random() > 0.35;
  return {
    passage_id: passageId,
    precondition_holds: holds,
    period: holds ? "01101" : null,
    period_length: holds ? 5 : null,
    classical_queries_needed: 32,
    quantum_queries_needed: holds ? 6 : null,
    speedup_ratio: holds ? parseFloat((32 / 6).toFixed(2)) : null,
    interpretation: holds
      ? `XOR-period structure detected at ${passageId}. Simon's algorithm recovers period in 6 queries vs 32 classical.`
      : `No XOR-period structure found at ${passageId}. Classical search is optimal for this passage.`,
    exit_code: 0,
  };
}

function mockHardness(nSamples, useQuantum) {
  const pgood = parseFloat((_rng(0.012, 0.048)).toFixed(4));
  const groverCalls = Math.round(Math.PI / (4 * Math.sqrt(pgood)));
  const classicalCalls = Math.round(1 / pgood);
  return {
    n_samples: nSamples,
    use_quantum_iqae: useQuantum,
    p_good: pgood,
    grover_oracle_calls: groverCalls,
    classical_expected_calls: classicalCalls,
    speedup_ratio: parseFloat((classicalCalls / groverCalls).toFixed(2)),
    thresholds: [
      { tau: 0.90, p_good: parseFloat((pgood * 0.62).toFixed(5)), grover_calls: Math.round(Math.PI / (4 * Math.sqrt(pgood * 0.62))) },
      { tau: 0.95, p_good: parseFloat((pgood * 0.41).toFixed(5)), grover_calls: Math.round(Math.PI / (4 * Math.sqrt(pgood * 0.41))) },
      { tau: 0.99, p_good: parseFloat((pgood * 0.18).toFixed(5)), grover_calls: Math.round(Math.PI / (4 * Math.sqrt(pgood * 0.18))) },
    ],
    recommendation: groverCalls < 50
      ? "QPU budget is justified: Grover speedup is " + (classicalCalls / groverCalls).toFixed(1) + "×."
      : "Classical MCMC is more practical at this hardness level.",
    exit_code: 0,
  };
}

const TOOL_DELAYS = {
  reconnaissance: [900, 1600], known_plaintext: [350, 650], crib_drag: [450, 900],
  oracle_probe: [1800, 3200], supply_chain_inject: [300, 550],
  query_history: [200, 420], run_mcmc_chain: [3200, 5500], declare_hypothesis: [500, 900],
  run_qaoa_subproblem: [2400, 4200], check_simon_period: [1200, 2200], measure_hardness: [800, 1600],
};

async function execMock(name, input, currentLm) {
  const [lo, hi] = TOOL_DELAYS[name] ?? [400, 800];
  await new Promise(r => setTimeout(r, _rng(lo, hi)));
  switch (name) {
    case "reconnaissance":      return mockRecon();
    case "known_plaintext":     return mockKnownPlaintext(input.min_confidence);
    case "crib_drag":           return mockCribDrag(input.sign_code, input.proposed_phoneme);
    case "oracle_probe":        return mockOracleProbe(input.sign_code, input.proposed_phoneme, currentLm);
    case "supply_chain_inject": return mockSupplyChain(input.top_k);
    case "query_history":       return mockQueryHistory();
    case "run_mcmc_chain":      return mockMCMC(input.focus_passage, currentLm);
    case "declare_hypothesis":  return mockDeclare(input);
    case "run_qaoa_subproblem": return mockQAOA(input.top_signs ?? 6, input.reps ?? 1, input.backend ?? "simulator", currentLm);
    case "check_simon_period":  return mockSimon(input.passage_id ?? "P007_ADHS");
    case "measure_hardness":    return mockHardness(input.n_samples ?? 1000, input.use_quantum_iqae ?? false);
    default:                    return { error: `Unknown tool: ${name}` };
  }
}

// ─── Anthropic API streaming ──────────────────────────────────────────────────

async function* parseSSE(response) {
  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (payload === "[DONE]") return;
      try { yield JSON.parse(payload); } catch { /* skip */ }
    }
  }
}

async function callClaude(messages, apiKey, onText) {
  const key = apiKey || (typeof window !== "undefined" && window.__ANTHROPIC_API_KEY__) || "";
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
      "x-api-key": key,
    },
    body: JSON.stringify({
      model: MODEL, max_tokens: 2048, stream: true,
      system: SYSTEM_PROMPT, tools: TOOL_DEFINITIONS, messages,
    }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API ${response.status}: ${text.slice(0, 200)}`);
  }

  const blocks  = [];
  let inputBuf  = "";
  let stopReason = null;

  for await (const ev of parseSSE(response)) {
    if (ev.type === "content_block_start") {
      const cb = ev.content_block;
      inputBuf = "";
      if (cb.type === "text")     blocks[ev.index] = { type: "text", text: "" };
      if (cb.type === "tool_use") blocks[ev.index] = { type: "tool_use", id: cb.id, name: cb.name, input: {} };
    } else if (ev.type === "content_block_delta") {
      const b = blocks[ev.index];
      if (!b) continue;
      if (ev.delta.type === "text_delta") {
        b.text = (b.text || "") + ev.delta.text;
        onText?.(b.text);
      } else if (ev.delta.type === "input_json_delta") {
        inputBuf += ev.delta.partial_json;
      }
    } else if (ev.type === "content_block_stop") {
      const b = blocks[ev.index];
      if (b?.type === "tool_use" && inputBuf) {
        try { b.input = JSON.parse(inputBuf); } catch { b.input = {}; }
      }
      inputBuf = "";
    } else if (ev.type === "message_delta") {
      stopReason = ev.delta?.stop_reason;
    }
  }
  return { content: blocks.filter(Boolean), stop_reason: stopReason };
}

// ─── Utility formatters ────────────────────────────────────────────────────────

function fmtInput(tool, inp) {
  if (!inp) return "—";
  if (tool === "crib_drag" || tool === "oracle_probe") return `${inp.sign_code} → ${inp.proposed_phoneme}`;
  if (tool === "supply_chain_inject") return `top_k = ${inp.top_k ?? 10}`;
  if (tool === "run_mcmc_chain") return inp.focus_passage ? `focus: ${inp.focus_passage}` : "full corpus";
  if (tool === "declare_hypothesis") {
    const n = Object.keys(inp.phoneme_map ?? {}).length;
    return `${n} signs, conf = ${((inp.confidence ?? 0) * 100).toFixed(0)}%`;
  }
  if (tool === "known_plaintext") return `min_conf = ${inp.min_confidence ?? 0.7}`;
  if (tool === "run_qaoa_subproblem") return `top_signs=${inp.top_signs ?? 6} reps=${inp.reps ?? 1} backend=${inp.backend ?? "sim"}`;
  if (tool === "check_simon_period") return `passage: ${inp.passage_id ?? "?"}` ;
  if (tool === "measure_hardness") return `n=${inp.n_samples ?? 1000} iqae=${inp.use_quantum_iqae ? "yes" : "no"}`;
  return "—";
}

function fmtResult(tool, res) {
  if (!res) return "no result";
  if (res.error) return `error: ${res.error}`;
  if (tool === "reconnaissance")
    return `IC ${res.index_of_coincidence?.odd_lines}, ${res.zipf_analysis?.n_types} types, best ${res.current_best?.overall_lm_score} nats`;
  if (tool === "known_plaintext")
    return `${res.n_candidates} candidates above threshold`;
  if (tool === "crib_drag")
    return `${res.corpus_occurrences} hits · ${res.parallel_passages_containing_sign} parallel passages`;
  if (tool === "oracle_probe") {
    const d = res.delta_lm;
    return `Δ${d >= 0 ? "+" : ""}${d} nats → ${d >= 0 ? "COMMIT" : "REJECT"}`;
  }
  if (tool === "supply_chain_inject")
    return `Hevesy rate ${(res.hevesy_recovery_rate * 100).toFixed(1)}% · top: ${res.top_pairs?.[0]?.rongo_code}→${res.top_pairs?.[0]?.proposed_indus_phoneme}`;
  if (tool === "query_history")
    return `${res.n_runs_found} runs · best ${res.best_runs?.[0]?.metrics?.best_lm_score} nats`;
  if (tool === "run_mcmc_chain")
    return `${res.overall_lm_score} nats · ${res.n_assignments} assignments · ${res.status}`;
  if (tool === "declare_hypothesis")
    return `${res.n_signs} signs · conf ${((res.confidence ?? 0) * 100).toFixed(0)}% · COMPLETE`;
  if (tool === "run_qaoa_subproblem") {
    const d = res.delta_vs_mcmc;
    return `Δ${d != null ? (d >= 0 ? "+" : "") + d.toFixed(2) : "?"} nats vs MCMC · ${res.best_lm_score} nats`;
  }
  if (tool === "check_simon_period")
    return res.precondition_holds
      ? `period found: s=${res.period} · ${res.quantum_queries_needed}q vs ${res.classical_queries_needed}c queries`
      : `no XOR-period · classical optimal`;
  if (tool === "measure_hardness")
    return `p_good=${res.p_good} · Grover=${res.grover_oracle_calls} calls · ${res.speedup_ratio}× speedup`;
  return JSON.stringify(res).slice(0, 90);
}

// ─── LogEntry component ───────────────────────────────────────────────────────

function LogEntry({ entry }) {
  const [open, setOpen] = useState(false);
  const color = TOOL_COLORS[entry.tool] ?? "#8b949e";
  return (
    <div style={{ borderLeft: `3px solid ${color}`, background: "#0d1117", borderRadius: 4,
                  padding: "8px 10px", marginBottom: 6, cursor: "pointer" }}
         onClick={() => setOpen(o => !o)}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
        <span style={{ color, border: `1px solid ${color}`, borderRadius: 3,
                       padding: "1px 5px", fontSize: 9, fontWeight: 700, letterSpacing: 1 }}>
          {entry.tool.replace(/_/g, " ").toUpperCase()}
        </span>
        <span style={{ color: "#8b949e", fontSize: 10 }}>T{entry.turn} · {entry.ts}</span>
        <span style={{ color: "#8b949e", fontSize: 10, marginLeft: "auto" }}>{entry.elapsed}s</span>
      </div>
      <div style={{ color: "#f0883e", fontSize: 11, marginBottom: 2 }}>{fmtInput(entry.tool, entry.input)}</div>
      <div style={{ color: "#c9d1d9", fontSize: 11 }}>{fmtResult(entry.tool, entry.result)}</div>
      {open && (
        <pre style={{ marginTop: 8, fontSize: 10, color: "#8b949e", background: "#161b22",
                      borderRadius: 3, padding: 8, overflowX: "auto", maxHeight: 220,
                      whiteSpace: "pre", lineHeight: 1.4 }}>
          {JSON.stringify(entry.result, null, 2)}
        </pre>
      )}
    </div>
  );
}

// ─── Phase badge ──────────────────────────────────────────────────────────────

const PHASES = {
  idle:    { label: "STANDBY",            color: "#8b949e" },
  reason:  { label: "AGENT REASONING",    color: "#58a6ff" },
  call:    { label: "EXECUTING TOOL",     color: "#f0883e" },
  done:    { label: "HYPOTHESIS DECLARED", color: "#ffd700" },
};
function PhaseBadge({ phase }) {
  const { label, color } = PHASES[phase] ?? PHASES.idle;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: color,
                     boxShadow: phase !== "idle" ? `0 0 6px ${color}` : "none",
                     display: "inline-block", flexShrink: 0 }} />
      <span style={{ color, fontWeight: 700, fontSize: 11, letterSpacing: 1 }}>{label}</span>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function RedRongoDashboard() {
  const [apiKey,    setApiKey]    = useState(() => (typeof window !== "undefined" && window.__ANTHROPIC_API_KEY__) || "");
  const [running,   setRunning]   = useState(false);
  const [entries,   setEntries]   = useState([]);
  const [lmData,    setLmData]    = useState(DEMO_INIT_LM_CHART);
  const [anchors,   setAnchors]   = useState(INIT_ANCHORS + DEMO_N_SOFT_PROMOTED);
  const [streamTxt, setStreamTxt] = useState("");
  const [phase,     setPhase]     = useState("idle");
  const [hypo,      setHypo]      = useState(null);
  const [err,       setErr]       = useState(null);
  const [qaoa,      setQaoa]      = useState(null);   // latest QAOA result
  const [simon,     setSimon]     = useState(null);   // latest Simon result
  const [hardness,  setHardness]  = useState(null);   // latest hardness result

  const logScrollRef = useRef(null);
  const abortRef     = useRef(false);

  // Scroll attack log to bottom on new entries
  useEffect(() => {
    if (logScrollRef.current) logScrollRef.current.scrollTop = logScrollRef.current.scrollHeight;
  }, [entries]);

  const run = useCallback(async () => {
    const key = apiKey.trim() || (typeof window !== "undefined" && window.__ANTHROPIC_API_KEY__) || "";
    if (!key) { setErr("Paste your Anthropic API key above."); return; }

    setErr(null); setRunning(true); setEntries([]); setStreamTxt("");
    // Reset chart to the real pre-run baseline from demo data (not a placeholder)
    setLmData(DEMO_INIT_LM_CHART);
    setAnchors(INIT_ANCHORS + DEMO_N_SOFT_PROMOTED);
    setHypo(null); setQaoa(null); setSimon(null); setHardness(null);
    setPhase("reason");
    abortRef.current = false; _mcmcCount = 0;

    const messages = [{
      role: "user",
      content: "Begin the attack. Execute a complete adversarial reasoning chain for the DEF CON AI Village audience.",
    }];

    let currentLm = DEMO_BEST_LM;
    let turn = 0;
    let done = false;

    try {
      while (turn < MAX_TURNS && !done && !abortRef.current) {
        turn++;
        setPhase("reason"); setStreamTxt("");

        const resp = await callClaude(messages, key, txt => setStreamTxt(txt));
        setStreamTxt("");
        messages.push({ role: "assistant", content: resp.content });

        if (resp.stop_reason === "end_turn") break;

        const toolBlocks = resp.content.filter(b => b.type === "tool_use");
        if (!toolBlocks.length) break;

        setPhase("call");
        const toolResults = [];

        for (const blk of toolBlocks) {
          if (abortRef.current) break;
          const t0 = Date.now();
          const result = await execMock(blk.name, blk.input, currentLm);
          const elapsed = ((Date.now() - t0) / 1000).toFixed(2);

          if (blk.name === "run_mcmc_chain" && result.overall_lm_score != null) {
            currentLm = result.overall_lm_score;
            setLmData(prev => [...prev, { label: `T${turn}`, score: Math.round(currentLm * 10) / 10 }]);
            setAnchors(prev => Math.min(prev + 2, TOTAL_SIGNS));
          }
          if (blk.name === "run_qaoa_subproblem" && result.best_lm_score != null) {
            setQaoa(result);
            // If QAOA beat MCMC, update chart and LM baseline
            if (result.delta_vs_mcmc > 0) {
              currentLm = result.best_lm_score;
              setLmData(prev => [...prev, { label: `Q${turn}`, score: Math.round(currentLm * 10) / 10 }]);
              setAnchors(prev => Math.min(prev + (result.qaoa_assignments?.length ?? 0), TOTAL_SIGNS));
            }
          }
          if (blk.name === "check_simon_period") setSimon(result);
          if (blk.name === "measure_hardness")   setHardness(result);
          if (blk.name === "oracle_probe" && result.delta_lm > 0) {
            setAnchors(prev => Math.min(prev + 1, TOTAL_SIGNS));
          }
          if (blk.name === "declare_hypothesis") {
            done = true;
            setHypo({
              phoneme_map:      blk.input.phoneme_map ?? {},
              attack_path:      blk.input.attack_path ?? [],
              evidence_summary: blk.input.evidence_summary ?? "",
              confidence:       blk.input.confidence ?? 0,
            });
            setAnchors(prev => Math.min(prev + Object.keys(blk.input.phoneme_map ?? {}).length, TOTAL_SIGNS));
          }

          setEntries(prev => [...prev, {
            id: `${turn}-${blk.name}-${Date.now()}`,
            turn, tool: blk.name, input: blk.input, result, elapsed,
            ts: new Date().toISOString().slice(11, 19),
          }]);

          toolResults.push({ type: "tool_result", tool_use_id: blk.id, content: JSON.stringify(result) });
        }
        messages.push({ role: "user", content: toolResults });
      }
    } catch (e) {
      setErr(e.message);
    } finally {
      setRunning(false); setStreamTxt("");
      setPhase(done ? "done" : "idle");
    }
  }, [apiKey]);

  const stop = useCallback(() => { abortRef.current = true; }, []);

  // Derived keyspace values
  const ksNow   = ksLog10(anchors);
  const ksStart = ksLog10(INIT_ANCHORS);
  const ksPct   = Math.max(0, Math.min(100, (ksNow / ksStart) * 100));
  const lastLm  = lmData[lmData.length - 1]?.score ?? -3287.4;
  const lmDelta = lastLm - lmData[0].score;

  return (
    <div style={{ background: "#0d1117", color: "#c9d1d9", fontFamily: "'JetBrains Mono', monospace",
                  height: "100vh", display: "flex", flexDirection: "column", fontSize: 12, overflow: "hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
      `}</style>

      {/* ── Header ── */}
      <div style={{ background: "#161b22", borderBottom: "1px solid #30363d",
                    padding: "8px 14px", display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", flexShrink: 0 }}>
        <span style={{ color: "#ff4444", fontWeight: 700, fontSize: 15, letterSpacing: 3 }}>REDRONGO</span>
        <span style={{ color: "#30363d", fontSize: 11 }}>|</span>
        <span style={{ color: "#8b949e", fontSize: 10, flex: 1 }}>
          Autonomous Rongorongo Cipher Attack · AI Village · DEF CON 34
        </span>
        <span style={{ background: "#21262d", border: "1px solid #30363d", borderRadius: 3,
                       color: "#8b949e", fontSize: 9, padding: "2px 6px", letterSpacing: 1 }}>
          DEMO MODE — tool results simulated
        </span>
        <input
          type="password"
          placeholder="sk-ant-... API key"
          value={apiKey}
          onChange={e => setApiKey(e.target.value)}
          disabled={running}
          style={{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 4,
                   color: "#c9d1d9", fontFamily: "inherit", fontSize: 11,
                   padding: "4px 8px", width: 220, outline: "none" }}
        />
        {!running
          ? <button onClick={run}
              style={{ background: "#ff4444", border: "none", borderRadius: 4, color: "#fff",
                       fontFamily: "inherit", fontWeight: 700, fontSize: 11, padding: "5px 14px",
                       cursor: "pointer", letterSpacing: 1 }}>
              LAUNCH ATTACK
            </button>
          : <button onClick={stop}
              style={{ background: "transparent", border: "1px solid #ff4444", borderRadius: 4,
                       color: "#ff4444", fontFamily: "inherit", fontWeight: 700, fontSize: 11,
                       padding: "5px 14px", cursor: "pointer", letterSpacing: 1 }}>
              ABORT
            </button>
        }
      </div>

      {err && (
        <div style={{ background: "#3d1515", borderBottom: "1px solid #ff4444",
                      color: "#ff8080", padding: "5px 14px", fontSize: 11, flexShrink: 0 }}>
          {err}
        </div>
      )}

      {/* ── Body ── */}
      <div style={{ display: "flex", flex: 1, gap: 8, padding: 8, overflow: "hidden" }}>

        {/* Left: Attack Log */}
        <div style={{ flex: 1, background: "#161b22", border: "1px solid #30363d",
                      borderRadius: 6, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div style={{ padding: "10px 12px 8px", borderBottom: "1px solid #21262d",
                        display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            <span style={{ color: "#39d353", fontWeight: 700, fontSize: 10, letterSpacing: 2 }}>ATTACK LOG</span>
            {running && <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#39d353",
                                       animation: "blink 1.2s infinite", display: "inline-block" }} />}
            <span style={{ color: "#30363d", fontSize: 10, marginLeft: "auto" }}>{entries.length} events</span>
          </div>

          <div ref={logScrollRef} style={{ flex: 1, overflowY: "auto", padding: "8px 10px 10px" }}>
            {/* Live reasoning */}
            {streamTxt && (
              <div style={{ background: "#0d1117", border: "1px solid #21262d", borderRadius: 4,
                             padding: "8px 10px", marginBottom: 8 }}>
                <span style={{ color: "#58a6ff", fontSize: 9, fontWeight: 700, letterSpacing: 1,
                               display: "block", marginBottom: 4 }}>AGENT REASONING</span>
                <div style={{ color: "#8b949e", fontSize: 11, lineHeight: 1.5,
                               maxHeight: 100, overflow: "hidden",
                               display: "-webkit-box", WebkitLineClamp: 5, WebkitBoxOrient: "vertical" }}>
                  {streamTxt}
                </div>
              </div>
            )}

            {entries.map(e => <LogEntry key={e.id} entry={e} />)}

            {entries.length === 0 && !running && (
              <div style={{ color: "#30363d", fontSize: 11, textAlign: "center", padding: "48px 0" }}>
                Awaiting attack launch. Tool calls and results will appear here.
              </div>
            )}

            {/* Hypothesis declaration */}
            {hypo && (
              <div style={{ background: "#1a1600", border: "1px solid #ffd700",
                             borderRadius: 5, padding: 12, marginTop: 8 }}>
                <div style={{ color: "#ffd700", fontWeight: 700, fontSize: 10,
                               letterSpacing: 2, marginBottom: 6 }}>HYPOTHESIS DECLARED</div>
                <div style={{ color: "#c9d1d9", fontSize: 12, marginBottom: 8 }}>
                  Confidence: {(hypo.confidence * 100).toFixed(0)}%
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
                  {Object.entries(hypo.phoneme_map).map(([k, v]) => (
                    <span key={k} style={{ background: "#21262d", border: "1px solid #30363d",
                                           borderRadius: 3, padding: "2px 6px",
                                           fontSize: 11, color: "#39d353" }}>
                      {k} → {v}
                    </span>
                  ))}
                </div>
                {hypo.attack_path.length > 0 && (
                  <div style={{ color: "#8b949e", fontSize: 10, marginBottom: 6 }}>
                    {hypo.attack_path.join(" → ")}
                  </div>
                )}
                <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: 13,
                               color: "#8b949e", lineHeight: 1.6, fontStyle: "italic" }}>
                  {hypo.evidence_summary}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right column */}
        <div style={{ width: 300, display: "flex", flexDirection: "column", gap: 8 }}>

          {/* Keyspace Tracker */}
          <div style={{ background: "#161b22", border: "1px solid #30363d",
                         borderRadius: 6, padding: 12 }}>
            <div style={{ color: "#39d353", fontWeight: 700, fontSize: 10,
                           letterSpacing: 2, borderBottom: "1px solid #21262d",
                           paddingBottom: 7, marginBottom: 10 }}>KEYSPACE TRACKER</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: "#ff4444",
                           marginBottom: 2, lineHeight: 1 }}>
              10<sup style={{ fontSize: 14 }}>{ksNow.toFixed(1)}</sup>
            </div>
            <div style={{ fontSize: 10, color: "#8b949e", marginBottom: 10 }}>
              {BASE}^{TOTAL_SIGNS - anchors} possible assignments
            </div>
            <div style={{ background: "#21262d", borderRadius: 3, height: 10, overflow: "hidden", marginBottom: 7 }}>
              <div style={{ background: "linear-gradient(90deg, #ff4444, #f0883e)",
                             height: "100%", borderRadius: 3, width: `${ksPct}%`,
                             transition: "width 0.8s ease" }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 4 }}>
              <span style={{ color: "#39d353" }}>{anchors} anchored</span>
              <span style={{ color: "#8b949e" }}>{TOTAL_SIGNS - anchors} unanchored</span>
            </div>
            <div style={{ fontSize: 9, color: "#30363d" }}>
              baseline 10^{ksStart.toFixed(1)} · {INIT_ANCHORS} hard anchors at launch
            </div>
          </div>

          {/* LM Score Trajectory */}
          <div style={{ background: "#161b22", border: "1px solid #30363d",
                         borderRadius: 6, padding: 12, flex: 1 }}>
            <div style={{ color: "#39d353", fontWeight: 700, fontSize: 10,
                           letterSpacing: 2, borderBottom: "1px solid #21262d",
                           paddingBottom: 7, marginBottom: 10 }}>LM SCORE TRAJECTORY</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: "#39d353",
                           marginBottom: 2, lineHeight: 1 }}>
              {lastLm.toFixed(1)}
            </div>
            <div style={{ fontSize: 10, color: "#8b949e", marginBottom: 10 }}>
              {lmData.length > 1
                ? `${lmDelta >= 0 ? "+" : ""}${lmDelta.toFixed(1)} nats from baseline`
                : "awaiting first MCMC run"}
            </div>
            <ResponsiveContainer width="100%" height={130}>
              <LineChart data={lmData} margin={{ top: 4, right: 6, left: -24, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                <XAxis dataKey="label"
                  tick={{ fill: "#8b949e", fontSize: 9, fontFamily: "JetBrains Mono" }}
                  axisLine={{ stroke: "#30363d" }} tickLine={false} />
                <YAxis
                  tick={{ fill: "#8b949e", fontSize: 9, fontFamily: "JetBrains Mono" }}
                  axisLine={{ stroke: "#30363d" }} tickLine={false} domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{ background: "#161b22", border: "1px solid #30363d",
                                  borderRadius: 4, fontFamily: "JetBrains Mono", fontSize: 10,
                                  color: "#c9d1d9" }}
                  formatter={v => [`${v} nats`, "LM score"]} />
                <Line type="monotone" dataKey="score" stroke="#39d353" strokeWidth={2}
                  dot={({ cx, cy, payload }) => (
                    <circle key={payload.label} cx={cx} cy={cy} r={3} strokeWidth={0}
                      fill={payload.label?.startsWith("Q") ? "#a5f3fc" : "#39d353"} />
                  )}
                  activeDot={{ r: 4, fill: "#39d353" }} />
              </LineChart>
            </ResponsiveContainer>
            {/* Q-prefix dot legend */}
            {lmData.some(d => d.label?.startsWith("Q")) && (
              <div style={{ fontSize: 9, color: "#8b949e", marginTop: 4, display: "flex", gap: 10 }}>
                <span><span style={{ color: "#39d353" }}>●</span> MCMC</span>
                <span><span style={{ color: "#a5f3fc" }}>●</span> QAOA</span>
              </div>
            )}
          </div>

          {/* Quantum Results Panel */}
          {(qaoa || simon || hardness) && (
            <div style={{ background: "#161b22", border: "1px solid #30363d",
                           borderRadius: 6, padding: 10 }}>
              <div style={{ color: "#a5f3fc", fontWeight: 700, fontSize: 10,
                             letterSpacing: 2, borderBottom: "1px solid #21262d",
                             paddingBottom: 6, marginBottom: 8 }}>QUANTUM RESULTS</div>

              {qaoa && (
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 9, color: "#8b949e", letterSpacing: 1, marginBottom: 3 }}>QAOA SUBPROBLEM</div>
                  <div style={{ display: "flex", gap: 10 }}>
                    <div>
                      <div style={{ fontSize: 11, color: qaoa.delta_vs_mcmc >= 0 ? "#39d353" : "#f87171", fontWeight: 700 }}>
                        {qaoa.delta_vs_mcmc >= 0 ? "+" : ""}{qaoa.delta_vs_mcmc?.toFixed(2)} nats
                      </div>
                      <div style={{ fontSize: 9, color: "#8b949e" }}>vs MCMC</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: "#a5f3fc", fontWeight: 700 }}>{qaoa.best_lm_score?.toFixed(1)}</div>
                      <div style={{ fontSize: 9, color: "#8b949e" }}>LM score</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: "#c9d1d9" }}>{qaoa.n_function_evaluations}</div>
                      <div style={{ fontSize: 9, color: "#8b949e" }}>fn evals</div>
                    </div>
                  </div>
                </div>
              )}

              {simon && (
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 9, color: "#8b949e", letterSpacing: 1, marginBottom: 3 }}>SIMON'S ALGORITHM</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{
                      background: simon.precondition_holds ? "rgba(74,222,128,.12)" : "rgba(248,113,113,.1)",
                      color: simon.precondition_holds ? "#39d353" : "#f87171",
                      fontSize: 9, padding: "1px 5px", borderRadius: 3,
                    }}>
                      {simon.precondition_holds ? "XOR-PERIOD FOUND" : "NO PERIOD"}
                    </span>
                    {simon.precondition_holds && (
                      <span style={{ fontSize: 10, color: "#f9a8d4" }}>s={simon.period}</span>
                    )}
                    {simon.speedup_ratio && (
                      <span style={{ fontSize: 10, color: "#8b949e" }}>{simon.speedup_ratio}× speedup</span>
                    )}
                  </div>
                </div>
              )}

              {hardness && (
                <div>
                  <div style={{ fontSize: 9, color: "#8b949e", letterSpacing: 1, marginBottom: 3 }}>HARDNESS CERTIFICATE</div>
                  <div style={{ display: "flex", gap: 10 }}>
                    <div>
                      <div style={{ fontSize: 11, color: "#fde68a", fontWeight: 700 }}>{hardness.p_good}</div>
                      <div style={{ fontSize: 9, color: "#8b949e" }}>p_good</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: "#fde68a", fontWeight: 700 }}>{hardness.grover_oracle_calls}</div>
                      <div style={{ fontSize: 9, color: "#8b949e" }}>Grover calls</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: "#c9d1d9", fontWeight: 700 }}>{hardness.speedup_ratio}×</div>
                      <div style={{ fontSize: 9, color: "#8b949e" }}>speedup</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 9, color: "#8b949e", marginTop: 4, fontStyle: "italic" }}>
                    {hardness.recommendation}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Phase indicator */}
          <div style={{ background: "#161b22", border: "1px solid #30363d",
                         borderRadius: 6, padding: "10px 12px" }}>
            <PhaseBadge phase={phase} />
          </div>
        </div>
      </div>
    </div>
  );
}
