import { useCallback, useEffect, useState } from "react";
import { Activity, ArrowUpRight, Atom, CheckCircle2, CircleHelp, Command, FlaskConical, Layers3, LoaderCircle, Network, RefreshCw, Search, ShieldCheck, Sparkles, TerminalSquare, XCircle, Clock3, ListChecks } from "lucide-react";

const API_URL = (import.meta.env.VITE_AURA_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
type Health = { status: string; service?: string };
type ResearchTask = { task_id: string; task_type?: string; description?: string; status?: string; dependencies?: string[] };
type ResearchPlan = { question?: string; objectives?: string[]; hypotheses?: string[]; success_criteria?: string[] };
type ResearchRun = { run_id: string; status?: string; research_question: string; tenant_id?: string | null; tasks?: string[]; completed_tasks?: string[]; failed_tasks?: string[] };
type CreatedRun = { run: ResearchRun; plan: ResearchPlan; tasks: ResearchTask[] };\ntype ResearchJob = { job_id: string; run_id: string; status: string; progress: number; completed_tasks: number; total_tasks: number; attempts: number; max_attempts: number; error?: string | null };\ntype JobAttempt = { attempt_id: string; attempt_number: number; worker_id: string; status: string; started_at: string; finished_at?: string | null; error?: string | null; error_type?: string | null };\ntype JobDiagnostics = { job_id: string; status: string; error?: string | null; attempts: JobAttempt[]; tasks: { task_id: string; status: string; checkpoint_status?: string | null; checkpoint_error?: string | null }[] };
type View = "overview" | "research" | "pipeline";

async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "AURA API request failed.");
  return payload as T;
}
function statusClass(status?: string) { return (status || "unknown").toLowerCase().replace(/[^a-z0-9_-]/g, "-"); }

function App() {
  const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
  const [question, setQuestion] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const [created, setCreated] = useState<CreatedRun | null>(null);
  const [view, setView] = useState<View>("overview");
  const [runs, setRuns] = useState<ResearchRun[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState("");
  const [search, setSearch] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [tasks, setTasks] = useState<ResearchTask[]>([]);
  const [tasksLoading, setTasksLoading] = useState(false);
  const [tasksError, setTasksError] = useState("");\n  const [job, setJob] = useState<ResearchJob | null>(null);\n  const [diagnostics, setDiagnostics] = useState<JobDiagnostics | null>(null);\n  const [executionBusy, setExecutionBusy] = useState(false);\n  const [executionError, setExecutionError] = useState("");

  const checkHealth = useCallback(async () => {
    setHealth("checking");
    try { const payload = await apiGet<Health>("/health"); setHealth(payload.status === "ok" ? "online" : "offline"); }
    catch { setHealth("offline"); }
  }, []);
  const loadRuns = useCallback(async () => {
    setRunsLoading(true); setRunsError("");
    try {
      const payload = await apiGet<ResearchRun[]>("/research/runs");
      setRuns(payload);
      setSelectedRunId((current) => current || payload[0]?.run_id || "");
    } catch (caught) { setRunsError(caught instanceof Error ? caught.message : "Could not load research runs."); }
    finally { setRunsLoading(false); }
  }, []);
  const loadTasks = useCallback(async (runId: string) => {
    if (!runId) { setTasks([]); return; }
    setTasksLoading(true); setTasksError("");
    try { setTasks(await apiGet<ResearchTask[]>(`/research/runs/${encodeURIComponent(runId)}/tasks`)); }
    catch (caught) { setTasksError(caught instanceof Error ? caught.message : "Could not load the task graph."); setTasks([]); }
    finally { setTasksLoading(false); }
  }, []);

  useEffect(() => { void checkHealth(); void loadRuns(); }, [checkHealth, loadRuns]);
  useEffect(() => { if (view === "research") void loadRuns(); }, [view, loadRuns]);
  useEffect(() => { if (view === "pipeline" && selectedRunId) void loadTasks(selectedRunId); }, [view, selectedRunId, loadTasks]);\n  useEffect(() => { if (view !== "pipeline" || !job || !["queued", "running", "cancel_requested"].includes(job.status)) return; const timer = window.setInterval(() => { void refreshExecution(job.job_id); }, 2500); return () => window.clearInterval(timer); }, [view, job?.job_id, job?.status, refreshExecution]);

  const selectedRun = runs.find((run) => run.run_id === selectedRunId) || null;\n  const refreshExecution = useCallback(async (jobId: string) => {\n    try {\n      const [nextJob, nextDiagnostics] = await Promise.all([apiGet<ResearchJob>(`/research/jobs/${encodeURIComponent(jobId)}`), apiGet<JobDiagnostics>(`/research/jobs/${encodeURIComponent(jobId)}/diagnostics`)]);\n      setJob(nextJob); setDiagnostics(nextDiagnostics);\n      await loadTasks(nextJob.run_id);\n    } catch (caught) { setExecutionError(caught instanceof Error ? caught.message : "Could not refresh execution status."); }\n  }, [loadTasks]);\n  const startExecution = async () => {\n    if (!selectedRunId || executionBusy) return;\n    setExecutionBusy(true); setExecutionError("");\n    try { const next = await apiGet<ResearchJob>(`/research/runs/${encodeURIComponent(selectedRunId)}/jobs`.replace("/jobs", "/jobs")); setJob(next); setDiagnostics(null); await refreshExecution(next.job_id); }\n    catch (caught) { setExecutionError(caught instanceof Error ? caught.message : "Could not start execution. Ensure a background worker is running."); }\n    finally { setExecutionBusy(false); }\n  };\n  const pauseExecution = async () => {\n    if (!job || executionBusy) return;\n    setExecutionBusy(true); setExecutionError("");\n    try { const response = await fetch(`${API_URL}/research/jobs/${encodeURIComponent(job.job_id)}/cancel`, { method: "POST" }); const payload = await response.json(); if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Could not stop execution."); setJob(payload as ResearchJob); await refreshExecution(job.job_id); }\n    catch (caught) { setExecutionError(caught instanceof Error ? caught.message : "Could not stop execution."); }\n    finally { setExecutionBusy(false); }\n  };\n  const retryExecution = async () => {\n    if (!job || executionBusy) return;\n    setExecutionBusy(true); setExecutionError("");\n    try { const response = await fetch(`${API_URL}/research/jobs/${encodeURIComponent(job.job_id)}/retry`, { method: "POST" }); const payload = await response.json(); if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Could not retry execution."); setJob(payload as ResearchJob); await refreshExecution(job.job_id); }\n    catch (caught) { setExecutionError(caught instanceof Error ? caught.message : "Could not retry execution."); }\n    finally { setExecutionBusy(false); }\n  };
  const filteredRuns = runs.filter((run) => run.research_question.toLowerCase().includes(search.toLowerCase()) || run.run_id.toLowerCase().includes(search.toLowerCase()));

  async function startResearch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion || creating) return;
    setCreating(true); setError(""); setCreated(null);
    try {
      const response = await fetch(`${API_URL}/research/runs`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: cleanQuestion }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Unable to create research run.");
      const result = payload as CreatedRun;
      setCreated(result);
      setRuns((current) => [result.run, ...current.filter((run) => run.run_id !== result.run.run_id)]);
      setSelectedRunId(result.run.run_id); setQuestion("");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Something went wrong while contacting AURA."); }
    finally { setCreating(false); }
  }

  const pageTitle = view === "overview" ? "Overview" : view === "research" ? "Research runs" : "Agent pipeline";
  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#overview" onClick={() => setView("overview")} aria-label="AURA home"><span className="brand-mark"><Atom size={22} strokeWidth={1.8} /></span><span><strong>AURA</strong><small>RESEARCH SYSTEM</small></span></a>
      <div className="workspace-label">WORKSPACE</div>
      <nav className="nav-list" aria-label="Main navigation">
        <button className={`nav-item ${view === "overview" ? "active" : ""}`} onClick={() => setView("overview")}><Layers3 size={17} /> Overview</button>
        <button className={`nav-item ${view === "research" ? "active" : ""}`} onClick={() => setView("research")}><FlaskConical size={17} /> Research runs</button>
        <button className={`nav-item ${view === "pipeline" ? "active" : ""}`} onClick={() => setView("pipeline")}><Network size={17} /> Agent pipeline</button>
      </nav>
      <div className="sidebar-bottom"><div className="system-card"><div className="system-card-title"><ShieldCheck size={15} /> SYSTEM STATUS</div><div className="system-status"><span className={`status-light ${health}`} />{health === "online" ? "API connected" : health === "checking" ? "Checking API…" : "API unavailable"}</div><button className="text-button" onClick={() => void checkHealth()}><RefreshCw size={12} /> Check connection</button></div><div className="sidebar-foot"><span className="tiny-orb" /> AUTONOMOUS BY DESIGN</div></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div className="breadcrumbs"><span>Workspace</span><span className="crumb-sep">/</span><strong>{pageTitle}</strong></div><div className="topbar-right"><span className="environment"><span className="status-light online" /> LOCAL ENVIRONMENT</span><span className="avatar">A</span></div></header>
      <div className="content-wrap">
        {view === "overview" && <>
          <section className="welcome-row"><div><div className="eyebrow"><Sparkles size={13} /> AUTONOMOUS RESEARCH WORKSPACE</div><h1>Turn questions into <span>discoveries.</span></h1><p className="welcome-copy">Give AURA a research question. It will shape a plan, define tasks, and prepare a reproducible investigation.</p></div><div className="welcome-decoration" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit-core"><Atom size={31} /></div></div></section>
          <section className="metrics-grid" aria-label="System overview">
            <article className="metric-card"><div className="metric-top"><span>API STATUS</span><Activity size={16} /></div><div className="metric-value status-value">{health === "online" ? "Operational" : health === "checking" ? "Checking…" : "Disconnected"}</div><div className="metric-foot"><span className={`status-light ${health}`} /> {API_URL}</div></article>
            <article className="metric-card"><div className="metric-top"><span>RESEARCH RUNS</span><Clock3 size={16} /></div><div className="metric-value">{runs.length.toString().padStart(2, "0")}</div><div className="metric-foot">Persisted runs for this tenant</div></article>
            <article className="metric-card"><div className="metric-top"><span>RESEARCH ENGINE</span><Command size={16} /></div><div className="metric-value">AURA Core</div><div className="metric-foot"><span className="accent-dot" /> Planning and orchestration</div></article>
          </section>
          <section className="research-layout">
            <div className="panel composer-panel"><div className="panel-heading"><div><div className="eyebrow">NEW INVESTIGATION</div><h2>What do you want to know?</h2></div><span className="step-chip">01 <span>/ 03</span></span></div><p className="panel-copy">Describe a focused question. AURA will generate a research plan and task graph before execution.</p>
              <form onSubmit={startResearch}><label className="sr-only" htmlFor="research-question">Research question</label><textarea id="research-question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="e.g. How does retrieval chunk size affect answer accuracy and latency in a RAG system?" maxLength={4000} rows={5} required /><div className="composer-meta"><span>{question.length}/4000 characters</span><span><CircleHelp size={13} /> Be specific for better results</span></div>{error && <div className="error-message"><XCircle size={15} /> {error}</div>}<div className="composer-actions"><span className="secure-note"><ShieldCheck size={14} /> Sent to your configured AURA API</span><button className="primary-button" type="submit" disabled={!question.trim() || creating || health !== "online"}>{creating ? <><LoaderCircle className="spin" size={16} /> Creating plan…</> : <>Create research run <ArrowUpRight size={16} /></>}</button></div></form>
            </div>
            <aside className="panel pipeline-panel"><div className="eyebrow">HOW AURA WORKS</div><h2>From question to evidence.</h2><div className="pipeline-list">{["Understand|Parse the question and objectives","Plan|Build hypotheses and task graph","Investigate|Gather evidence and run experiments","Synthesize|Evaluate and report findings"].map((step, index) => { const [title, description] = step.split("|"); return <div className="pipeline-step" key={title}><span className="pipeline-icon">{index === 0 ? <Search size={16} /> : index === 1 ? <Network size={16} /> : index === 2 ? <FlaskConical size={16} /> : <CheckCircle2 size={16} />}</span><div><strong>{title}</strong><small>{description}</small></div><span className="step-number">{String(index + 1).padStart(2, "0")}</span></div>; })}</div><div className="pipeline-note"><TerminalSquare size={14} /> This milestone adds persisted run history and task-graph inspection. Execution is queued through the API and processed by a separate background worker.</div></aside>
          </section>
          {created && <section className="panel result-panel" aria-live="polite"><div className="result-heading"><div><div className="eyebrow"><CheckCircle2 size={13} /> RESEARCH RUN CREATED</div><h2>Your investigation is planned.</h2></div><span className="run-id">{created.run.run_id}</span></div><p className="result-question">{created.plan.question || created.run.research_question}</p><div className="result-columns"><div><h3>Objectives</h3>{created.plan.objectives?.length ? <ul>{created.plan.objectives.map((item, index) => <li key={index}>{item}</li>)}</ul> : <p className="muted">No objectives returned.</p>}</div><div><h3>Task graph <span className="count-pill">{created.tasks.length}</span></h3>{created.tasks.length ? <ul>{created.tasks.map((task) => <li key={task.task_id}><strong>{task.task_type || "Task"}</strong> — {task.description || task.task_id}</li>)}</ul> : <p className="muted">No tasks returned.</p>}</div></div><button className="secondary-button" onClick={() => setView("pipeline")}>Inspect agent pipeline <ArrowUpRight size={14} /></button></section>}
        </>}
        {view === "research" && <section className="page-section">
          <div className="page-heading"><div><div className="eyebrow">RESEARCH WORKSPACE</div><h1>Research runs</h1><p className="welcome-copy">Browse persisted investigations and open their task graphs.</p></div><button className="secondary-button" onClick={() => void loadRuns()} disabled={runsLoading}><RefreshCw size={14} /> Refresh</button></div>
          <div className="run-toolbar"><div className="search-field"><Search size={16} /><input aria-label="Search research runs" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search by question or run ID" /></div><span className="run-count">{runs.length} total</span></div>
          {runsError && <div className="error-message"><XCircle size={15} /> {runsError}</div>}
          {runsLoading ? <div className="empty-state"><LoaderCircle className="spin" /> Loading research runs…</div> : filteredRuns.length ? <div className="run-list">{filteredRuns.map((run) => <button className="run-card" key={run.run_id} onClick={() => { setSelectedRunId(run.run_id); setView("pipeline"); }}><div className="run-card-top"><span className={`status-pill ${statusClass(run.status)}`}>{run.status || "unknown"}</span><span className="run-card-id">{run.run_id}</span></div><strong>{run.research_question}</strong><div className="run-card-foot"><span><ListChecks size={14} /> {(run.tasks || []).length} tasks</span><span>{(run.completed_tasks || []).length} completed</span><span className="open-run">Open pipeline <ArrowUpRight size={13} /></span></div></button>)}</div> : <div className="empty-state"><FlaskConical size={25} /><strong>{search ? "No matching research runs" : "No research runs yet"}</strong><span>{search ? "Try another search term." : "Create an investigation from Overview to see it here."}</span></div>}
        </section>}
        {view === "pipeline" && <section className="page-section">
          <div className="page-heading"><div><div className="eyebrow">AUTONOMOUS EXECUTION</div><h1>Agent pipeline</h1><p className="welcome-copy">Launch a durable background job, monitor task progress, inspect worker diagnostics, and retry failed work.</p></div><button className="secondary-button" onClick={() => selectedRunId && void loadTasks(selectedRunId)} disabled={!selectedRunId || tasksLoading}><RefreshCw size={14} /> Refresh tasks</button></div>
          <div className="pipeline-toolbar"><label htmlFor="pipeline-run">Selected research run</label><select id="pipeline-run" value={selectedRunId} onChange={(event) => setSelectedRunId(event.target.value)}><option value="">Choose a research run</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.research_question.slice(0, 90)}</option>)}</select><button className="text-button" onClick={() => setView("research")}>Browse all runs <ArrowUpRight size={13} /></button></div>
          {tasksError && <div className="error-message"><XCircle size={15} /> {tasksError}</div>}\n          {selectedRunId && <section className="execution-panel"><div className="execution-heading"><div><div className="eyebrow">EXECUTION CONTROL</div><h2>{job ? `Job ${job.job_id.slice(0, 8)}` : "Ready to investigate"}</h2></div><div className="execution-actions">{!job || ["failed","cancelled","succeeded"].includes(job.status) ? <button className="primary-button" onClick={() => void startExecution()} disabled={executionBusy || !selectedRun || !["planned","blocked"].includes(selectedRun.status || "")}>{executionBusy ? "Working…" : "Start execution"}</button> : <button className="secondary-button" onClick={() => void pauseExecution()} disabled={executionBusy || !["queued","running"].includes(job.status)}>Pause / stop</button>}{job?.status === "failed" && <button className="secondary-button" onClick={() => void retryExecution()} disabled={executionBusy}>Retry failed work</button>}</div></div>{job && <><div className="execution-stats"><span className={`status-pill ${statusClass(job.status)}`}>{job.status.replace("_"," ")}</span><strong>{job.completed_tasks}/{job.total_tasks} tasks</strong><strong>{Math.round(job.progress)}%</strong><strong>Attempt {job.attempts}/{job.max_attempts}</strong></div><div className="progress-track"><span style={{width:`${Math.max(0,Math.min(100,job.progress))}%`}} /></div></>}{executionError && <div className="error-message"><XCircle size={15}/>{executionError}</div>}<p className="execution-hint">Background execution requires the AURA worker process. Status refreshes automatically while the job is queued or running.</p></section>}
          {!selectedRunId ? <div className="empty-state"><Network size={25} /><strong>Select a research run</strong><span>Choose a run above or create one from Overview.</span></div> : tasksLoading ? <div className="empty-state"><LoaderCircle className="spin" /> Loading task graph…</div> : <>
            <div className="pipeline-summary"><div><span>RUN STATUS</span><strong className={`status-pill ${statusClass(selectedRun?.status)}`}>{selectedRun?.status || "unknown"}</strong></div><div><span>TOTAL TASKS</span><strong>{tasks.length}</strong></div><div><span>COMPLETED</span><strong>{tasks.filter((task) => task.status === "completed").length}</strong></div><div><span>FAILED</span><strong>{tasks.filter((task) => task.status === "failed").length}</strong></div></div>
            <div className="task-list">{tasks.length ? tasks.map((task, index) => <article className="task-card" key={task.task_id}><div className="task-index">{String(index + 1).padStart(2, "0")}</div><div className="task-main"><div className="task-card-top"><strong>{task.task_type || "Task"}</strong><span className={`status-pill ${statusClass(task.status)}`}>{task.status || "pending"}</span></div><p>{task.description}</p><div className="task-meta"><span>ID: {task.task_id}</span><span>Depends on: {task.dependencies?.length ? task.dependencies.join(", ") : "None"}</span>{task.status === "failed" && job?.status === "failed" && <button className="inline-retry" onClick={() => void retryExecution()} disabled={executionBusy}>Retry failed task</button>}</div></div></article>) : <div className="empty-state">This run has no persisted tasks.</div>}</div>
          </>}
        </section>}
        <section className="bottom-note"><div><span className="note-icon"><ShieldCheck size={16} /></span><span><strong>Research stays on your configured backend.</strong><small>This client talks directly to the AURA API URL shown above. Keep API credentials and provider secrets on the backend.</small></span></div><span className="version-label">CLIENT PREVIEW · 0.2.0</span></section>
      </div>
    </main>
  </div>;
}
export default App;
