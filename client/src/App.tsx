import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  Atom,
  CheckCircle2,
  CircleHelp,
  Command,
  FlaskConical,
  Layers3,
  LoaderCircle,
  Network,
  Plus,
  Radio,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  XCircle,
} from "lucide-react";

const API_URL = (import.meta.env.VITE_AURA_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

type Health = { status: string; service?: string };
type ResearchTask = {
  task_id: string;
  task_type?: string;
  description?: string;
  status?: string;
};
type ResearchPlan = {
  question?: string;
  objectives?: string[];
  hypotheses?: string[];
  success_criteria?: string[];
};
type ResearchRun = {
  run_id: string;
  status?: string;
  question?: string;
};
type CreatedRun = {
  run: ResearchRun;
  plan: ResearchPlan;
  tasks: ResearchTask[];
};

function App() {
  const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
  const [question, setQuestion] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const [created, setCreated] = useState<CreatedRun | null>(null);
  const [recentRuns, setRecentRuns] = useState<ResearchRun[]>([]);

  const checkHealth = useCallback(async () => {
    setHealth("checking");
    try {
      const response = await fetch(`${API_URL}/health`);
      if (!response.ok) throw new Error("API health check failed");
      const payload = (await response.json()) as Health;
      setHealth(payload.status === "ok" ? "online" : "offline");
    } catch {
      setHealth("offline");
    }
  }, []);

  useEffect(() => {
    void checkHealth();
  }, [checkHealth]);

  async function startResearch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion || creating) return;

    setCreating(true);
    setError("");
    setCreated(null);
    try {
      const response = await fetch(`${API_URL}/research/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: cleanQuestion }),
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = typeof payload.detail === "string" ? payload.detail : "Unable to create research run.";
        throw new Error(detail);
      }
      const result = payload as CreatedRun;
      setCreated(result);
      setRecentRuns((runs) => [result.run, ...runs.filter((run) => run.run_id !== result.run.run_id)].slice(0, 5));
      setQuestion("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong while contacting AURA.");
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="AURA home">
          <span className="brand-mark"><Atom size={22} strokeWidth={1.8} /></span>
          <span><strong>AURA</strong><small>RESEARCH SYSTEM</small></span>
        </a>

        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list" aria-label="Main navigation">
          <a className="nav-item active" href="#overview"><Layers3 size={17} /> Overview <span className="nav-dot" /></a>
          <a className="nav-item" href="#research"><FlaskConical size={17} /> Research runs</a>
          <a className="nav-item" href="#pipeline"><Network size={17} /> Agent pipeline</a>
        </nav>

        <div className="sidebar-bottom">
          <div className="system-card">
            <div className="system-card-title"><ShieldCheck size={15} /> SYSTEM STATUS</div>
            <div className="system-status">
              <span className={`status-light ${health}`} />
              {health === "online" ? "API connected" : health === "checking" ? "Checking API…" : "API unavailable"}
            </div>
            <button className="text-button" onClick={() => void checkHealth()}><RefreshCw size={12} /> Check connection</button>
          </div>
          <div className="sidebar-foot"><span className="tiny-orb" /> AUTONOMOUS BY DESIGN</div>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div className="breadcrumbs"><span>Workspace</span><span className="crumb-sep">/</span><strong>Overview</strong></div>
          <div className="topbar-right">
            <span className="environment"><span className="status-light online" /> LOCAL ENVIRONMENT</span>
            <span className="avatar">A</span>
          </div>
        </header>

        <div className="content-wrap">
          <section className="welcome-row">
            <div>
              <div className="eyebrow"><Sparkles size={13} /> AUTONOMOUS RESEARCH WORKSPACE</div>
              <h1>Turn questions into <span>discoveries.</span></h1>
              <p className="welcome-copy">Give AURA a research question. It will shape a plan, define tasks, and prepare a reproducible investigation.</p>
            </div>
            <div className="welcome-decoration" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit-core"><Atom size={31} /></div></div>
          </section>

          <section className="metrics-grid" aria-label="System overview">
            <article className="metric-card">
              <div className="metric-top"><span>API STATUS</span><Activity size={16} /></div>
              <div className="metric-value status-value">{health === "online" ? "Operational" : health === "checking" ? "Checking…" : "Disconnected"}</div>
              <div className="metric-foot"><span className={`status-light ${health}`} /> {API_URL}</div>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span>ACTIVE SESSION</span><Radio size={16} /></div>
              <div className="metric-value">{recentRuns.length.toString().padStart(2, "0")}</div>
              <div className="metric-foot">Runs created in this session</div>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span>RESEARCH ENGINE</span><Command size={16} /></div>
              <div className="metric-value">AURA Core</div>
              <div className="metric-foot"><span className="accent-dot" /> Planning and orchestration</div>
            </article>
          </section>

          <section className="research-layout" id="research">
            <div className="panel composer-panel">
              <div className="panel-heading">
                <div><div className="eyebrow">NEW INVESTIGATION</div><h2>What do you want to know?</h2></div>
                <span className="step-chip">01 <span>/ 03</span></span>
              </div>
              <p className="panel-copy">Describe a focused question. AURA will generate a research plan and task graph before execution.</p>
              <form onSubmit={startResearch}>
                <label className="sr-only" htmlFor="research-question">Research question</label>
                <textarea
                  id="research-question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="e.g. How does retrieval chunk size affect answer accuracy and latency in a RAG system?"
                  maxLength={4000}
                  rows={5}
                  required
                />
                <div className="composer-meta"><span>{question.length}/4000 characters</span><span><CircleHelp size={13} /> Be specific for better results</span></div>
                {error && <div className="error-message"><XCircle size={15} /> {error}</div>}
                <div className="composer-actions">
                  <span className="secure-note"><ShieldCheck size={14} /> Sent to your configured AURA API</span>
                  <button className="primary-button" type="submit" disabled={!question.trim() || creating || health !== "online"}>
                    {creating ? <><LoaderCircle className="spin" size={16} /> Creating plan…</> : <>Create research run <ArrowUpRight size={16} /></>}
                  </button>
                </div>
              </form>
            </div>

            <aside className="panel pipeline-panel" id="pipeline">
              <div className="eyebrow">HOW AURA WORKS</div>
              <h2>From question to evidence.</h2>
              <div className="pipeline-list">
                <div className="pipeline-step"><span className="pipeline-icon"><Search size={16} /></span><div><strong>Understand</strong><small>Parse the question and objectives</small></div><span className="step-number">01</span></div>
                <div className="pipeline-line" />
                <div className="pipeline-step"><span className="pipeline-icon"><Network size={16} /></span><div><strong>Plan</strong><small>Build hypotheses and task graph</small></div><span className="step-number">02</span></div>
                <div className="pipeline-line" />
                <div className="pipeline-step"><span className="pipeline-icon"><FlaskConical size={16} /></span><div><strong>Investigate</strong><small>Gather evidence and run experiments</small></div><span className="step-number">03</span></div>
                <div className="pipeline-line" />
                <div className="pipeline-step"><span className="pipeline-icon"><CheckCircle2 size={16} /></span><div><strong>Synthesize</strong><small>Evaluate and report findings</small></div><span className="step-number">04</span></div>
              </div>
              <div className="pipeline-note"><TerminalSquare size={14} /> Planning is available from this workspace. Execution controls are coming in a later frontend milestone.</div>
            </aside>
          </section>

          {created && (
            <section className="panel result-panel" aria-live="polite">
              <div className="result-heading"><div><div className="eyebrow"><CheckCircle2 size={13} /> RESEARCH RUN CREATED</div><h2>Your investigation is planned.</h2></div><span className="run-id">{created.run.run_id}</span></div>
              <p className="result-question">{created.plan.question || created.run.question}</p>
              <div className="result-columns">
                <div><h3>Objectives</h3>{created.plan.objectives?.length ? <ul>{created.plan.objectives.map((item, index) => <li key={index}>{item}</li>)}</ul> : <p className="muted">No objectives returned.</p>}</div>
                <div><h3>Task graph <span className="count-pill">{created.tasks.length}</span></h3>{created.tasks.length ? <ul>{created.tasks.map((task) => <li key={task.task_id}><strong>{task.task_type || "Task"}</strong> — {task.description || task.task_id}</li>)}</ul> : <p className="muted">No tasks returned.</p>}</div>
              </div>
            </section>
          )}

          <section className="bottom-note">
            <div><span className="note-icon"><ShieldCheck size={16} /></span><span><strong>Research stays on your configured backend.</strong><small>This client talks directly to the AURA API URL shown above. Keep API credentials and provider secrets on the backend.</small></span></div>
            <span className="version-label">CLIENT PREVIEW · 0.1.0</span>
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
