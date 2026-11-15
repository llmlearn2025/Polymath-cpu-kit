// Packsmith UI — help topics, the help drawer, and the web-view editors for model-made HTML (plan, playbook) and
// approach records. Loaded before app.js.

// ------------------------------------------------------------------ help content
// Each topic: id, tab (where its "?" opens it), level (basic / intermediate / advanced), title, html.
const HELP = [
  // ---------------------------------------------------------------- start
  { id: "start", tab: "*", level: "basic", title: "What Packsmith does", html: `
<p>Packsmith turns documents and data into a <b>specialization pack</b>: a small set of extra expert layers trained for
your local base model (Qwen3-30B-A3B or OLMoE) and switched on or off in colibri in milliseconds. The base model is never changed.</p>
<p>A pack learns two things, and the best packs have both:</p>
<ul><li><b>Knowledge</b> — the facts, figures, definitions and rules in your sources.</li>
<li><b>Method</b> — how an expert in the field works through a problem: what to check, in what order, how to tell
look-alikes apart. This is what the <b>Approaches → Discover → Playbook</b> tabs extract, and what sets Packsmith apart
from tools that only turn text into question–answer pairs.</li></ul>
<p>Three ways in, from least to most control:</p>
<ol><li><b>Assistant</b> — say what you want learned from which document; it asks a few questions and runs everything.</li>
<li><b>★ Quick</b> (bottom bar) — one document or data file → a dataset in one step.</li>
<li><b>The tabs</b>, left to right — every step visible and editable.</li></ol>` },
  { id: "flow", tab: "*", level: "basic", title: "The steps, in order", html: `
<ol><li><b>Library</b>: upload documents; tag regions (chapters, table of contents, book index) with Mark start / Mark end (Help: <i>Region types and roles</i>).</li>
<li><b>Topics</b>: a topic is a set of regions from one or more documents (and curated data sets) that will become one pack.</li>
<li><b>Approaches</b> (optional): pick or draft how experts in this field read such texts.</li>
<li><b>Discover</b> (reasoning only): the model reads sample pages and proposes a mining plan; you edit and approve it.</li>
<li><b>Playbook</b> (reasoning only): every page is mapped against the plan and merged into the book's method; back-test it on the book's own worked examples.</li>
<li><b>Generate</b>: build a dataset — facts &amp; data, reasoning from a playbook, or both.</li>
<li><b>Datasets</b>: look through, search, edit and clean the items; continue unfinished runs.</li>
<li><b>Composer</b> (optional): combine datasets into one.</li>
<li><b>Train</b>: train the pack.</li>
<li><b>Workbench</b>: ask questions with and without the pack; evaluate on held-out tests.</li></ol>
<p>Facts-only datasets need no plan or playbook: Library → Topics → Generate → Train.</p>` },

  // ---------------------------------------------------------------- approaches
  { id: "approaches-basic", tab: "approaches", level: "basic", title: "Approaches: what a record is", html: `
<p>An <b>approach record</b> describes how experts in a domain work, so the model knows what to look for when it reads
your text. It is a starting point for Discover, not a rule: Discover adapts it to the actual book.</p>
<p>A record has:</p>
<ul><li><b>domain</b> and <b>intent</b> — e.g. “law / criminal offences”, intent <i>reasoning</i>. Intents:
reasoning (how a problem is worked through), craft (how an author writes), diagnostic (from signs to an identification),
procedure (how a task is done), mixed.</li>
<li><b>moves</b> — the steps experts take, each with a <i>pattern</i> from the universal list (element, distinguish,
exception, sequence, cue, …) and a description of what to extract.</li>
<li><b>situations</b> — where practitioners meet these problems (used to write realistic scenes).</li>
<li><b>scores</b> — back-test results of topics that used the record, so you can see what has worked.</li></ul>
<p><b>To use one:</b> open a record that matches your field; in Discover choose it (or press Suggest).</p>` },
  { id: "approaches-edit", tab: "approaches", level: "intermediate", title: "Approaches: editing in the web view", html: `
<p>The record opens in a <b>web view</b>: each move is a card with its name, pattern and description, editable in place.
Add or remove moves with the buttons on each card. <b>Text (JSON)</b> shows the same record as JSON for exact edits;
the two stay in step, and Save takes whichever view you are in.</p>
<p>Good moves are <i>verbs an expert would recognise</i>: “check each ingredient”, “distinguish from extortion”,
“apply the exceptions”, “order the evidence”. Keep 3–8 moves; more moves make mapping noisier.</p>` },
  { id: "approaches-draft", tab: "approaches", level: "advanced", title: "Approaches: drafting by analogy", html: `
<p><b>Draft</b> writes a record for a new field from its nearest existing records and the universal patterns — e.g.
“tax tribunal orders” borrows from offence commentary and policy analysis. Review it: rename moves into the field's own
words, delete moves the field does not make, and add situations from your practice. Save it; it then appears in Discover.</p>
<p>Records improve with use: each back-test score is written back to the record, so after a few topics you can tell
which records produce playbooks that solve the books' own examples.</p>` },

  // ---------------------------------------------------------------- discover
  { id: "discover-basic", tab: "discover", level: "basic", title: "Discover: the model proposes a mining plan", html: `
<p>Discover finds out <b>how this particular text reasons</b>. You give it a prompt; it reads pages in rounds (contents
first, then samples from your train regions) and after each round updates a <b>mining plan</b>: the moves the text makes,
what each looks like here, with cited examples.</p>
<ol><li>Choose a topic (top right).</li>
<li><b>1 · Approach</b>: pick a record (or Suggest).</li>
<li><b>2 · Prompt</b>: Build prompt, then read and edit it — add what you care about (“focus on how courts treat
electronic records”).</li>
<li><b>3 · Guided reading</b>: Send and read. The table shows each round: moves found, new ones, stability, coverage.</li>
<li>When it stops, review the plan on the right and <b>Approve plan</b>.</li></ol>` },
  { id: "discover-stop", tab: "discover", level: "intermediate", title: "Discover: the stopping rule and settings", html: `
<ul><li><b>pages/round</b> (6): more pages per round means fewer, larger calls. Keep the total under your text-per-call budget.</li>
<li><b>stable rounds</b> (2): stop when the set of moves has not changed for this many rounds.</li>
<li><b>coverage</b> (0.6): …and at least this share of the train regions has been sampled.</li>
<li><b>max pages</b> (120): a hard cap.</li></ul>
<p>If it never stabilises, the text probably mixes several kinds of reasoning: split it into two topics, or tighten the
prompt. If it stabilises on very few moves, the record may be too generic: edit the prompt to name the moves you expect.</p>` },
  { id: "discover-edit", tab: "discover", level: "intermediate", title: "Discover: editing the plan (web view vs HTML)", html: `
<p>The plan opens in the <b>web view</b>: moves are cards; click any text to edit it. The toolbar adds a move, adds an
extract line or example, changes a line's kind, or deletes the selected part. Formatting is kept valid automatically —
you cannot break the structure by a missing tag.</p>
<p><b>HTML</b> shows the raw markup for exact edits (rename a <code>data-name</code>, change a <code>data-pattern</code>).
A small mistake there (an unclosed tag) is repaired when you switch back to the web view.</p>
<p>Before approving: merge moves that are the same thing in different words, delete moves the text only mentions in
passing, and make sure each move has a clear <i>extract</i> line — mapping uses it to decide what to pull from each page.</p>` },
  { id: "discover-advanced", tab: "discover", level: "advanced", title: "Discover: advanced use", html: `
<ul><li><b>Several books in one topic</b>: Discover samples all documents; the plan should cover moves from each. The
alignment matrix in Playbook later shows which document makes which move.</li>
<li><b>Re-running</b>: each run starts a fresh draft; earlier rounds stay in the log. Approving a new plan makes mapping start afresh.</li>
<li><b>Model choice</b>: a larger model (e.g. 31B) gives tighter plans; reasoning models must have enough reply budget —
if replies come back empty, turn thinking off for that model in the LLM server.</li>
<li><b>Write your own plan</b>: you can skip Discover's reading and paste a plan in HTML, then approve it.</li></ul>` },

  // ---------------------------------------------------------------- playbook
  { id: "playbook-basic", tab: "playbook", level: "basic", title: "Playbook: the book's method, written down", html: `
<p>A <b>playbook</b> is an ordered list of <b>checks</b> an expert works through, each with branches (“if A and B → go to
check 4”), cues, exceptions, distinctions, facts and sources. It is built from the whole text, not a sample.</p>
<ol><li><b>Skeleton from contents + index</b> (optional): turns the tagged contents/index into a first outline.</li>
<li><b>Map regions</b>: every block of pages is matched against the plan; each match becomes a fragment
(condition, outcome, cue, exception, distinction, step, fact) with its page. Runs in parallel; resumable.</li>
<li><b>Build playbook</b>: per move, the fragments are merged into checks; contradictions between sources are flagged.</li>
<li>Read and edit it on the right; <b>Save playbook</b>.</li></ol>` },
  { id: "playbook-backtest", tab: "playbook", level: "intermediate", title: "Playbook: back-test", html: `
<p>The back-test finds the book's own <b>worked examples</b> (illustrations, case discussions, solved problems), hides
their conclusions, and asks the model to solve them using <b>only the playbook</b>. A judge compares the result with the
book's answer: match / partial / miss.</p>
<ul><li><b>train regions</b>: how well the playbook captures the method it was built from. Aim for 0.7+.</li>
<li><b>held-out regions</b>: tag a chapter as <i>heldout</i> in the Library; this is the honest score.</li>
<li><b>with playbook</b>: any playbook can be tested on this topic's examples — e.g. test a playbook built from book A on book B.</li></ul>
<p>Read the misses: each shows the book's conclusion and the playbook's. A miss usually means a missing exception or
distinction — add it in the playbook and save.</p>` },
  { id: "playbook-edit", tab: "playbook", level: "intermediate", title: "Playbook: editing (web view vs HTML)", html: `
<p>In the <b>web view</b> each check is a numbered card; each line shows its kind (branch, cue, exception, distinction,
fact, step, source, note). Click to edit text. The toolbar adds a check or a line, changes a line's kind, or deletes.
Branches keep their targets (<code>data-goto</code>) — edit those in HTML view if needed.</p>
<p><b>HTML</b> view is the stored format. If a hand edit leaves a tag unclosed, switching to the web view repairs it.</p>` },
  { id: "playbook-advanced", tab: "playbook", level: "advanced", title: "Playbook: strategies and reuse", html: `
<ul><li><b>Unified vs one per document</b> (Topics → strategy): unified merges all books into one playbook; per-document
also builds one per book, so you can compare how authors differ.</li>
<li><b>Alignment matrix</b>: moves × documents. Empty cells show a book that never makes a move — either a gap in the
book or a move the plan named badly.</li>
<li><b>Reuse</b>: a playbook is a method. In Generate, any playbook from any topic can drive generation on any text.</li>
<li><b>More ways to mine</b>: to make a pack produce the book's kind of output (drafts, reports, prose) or review work,
use <i>Generate → Extra mining</i>; it works with or without a playbook.</li>
<li><b>Built-in playbooks</b>: <i>fact reasoning</i> (works on any text's captured facts) and <i>fact book</i> (for
curated data) need no mining at all.</li>
<li><b>Mapping returns 0 fragments</b>: open <code>topics/&lt;id&gt;/fragments/raw/</code> to see what the model said;
check the model's loaded context and that thinking is not eating the reply budget.</li></ul>` },

  // ---------------------------------------------------------------- other tabs
  { id: "library", tab: "library", level: "basic", title: "Library: documents and regions", html: `
<p>Upload PDF, Word, EPUB, HTML or text. PDF pages keep their order in the file; others are split into pseudo-pages.
Scanned pages are read by the vision model (OCR this page / OCR pages without text).</p>
<p>Tag regions: go to the first page, <b>Mark start</b>, go to the last, <b>Mark end</b>, choose a type and role, Save.
Roles: <b>train</b> (learned from), <b>context</b> (table of contents / book index, used for structure), <b>heldout</b> (kept for the final test), <b>exclude</b>.
What each type and role does: <b>Region types and roles</b>.</p>` },
  { id: "regions", tab: "library", level: "basic", title: "Region types and roles: what is mined", html: `
<p>A region is a run of pages you tag. Its <b>type</b> says what the pages are; its <b>role</b> says how they are used.
Only <b>chapter</b>, <b>section</b> and <b>other</b> pages with the role <b>train</b> are mined. Everything else either
guides the mining or is kept out of it.</p>
<h4>Types</h4>
<table class="tbl"><tr><th>Type</th><th>What it is</th><th>Mined?</th><th>What it is used for</th></tr>
<tr><td><b>chapter</b>, <b>section</b>, <b>other</b></td><td>The body of the text: the pages that teach. The three behave
the same; the name is only a label for you.</td><td><b>yes</b> (role train)</td><td>Discovery samples them; Map regions,
facts &amp; data, worked examples (back-test) and extra mining read them; training items come from them.</td></tr>
<tr><td><b>table of contents</b></td><td>The book's list of parts, chapters and headings, with page numbers.</td><td>never</td>
<td>The outline for <b>Skeleton from table of contents + book index</b>; Discover reads its first 4 pages first, to see
the book's plan, and the rest after the chapters.</td></tr>
<tr><td><b>book index</b></td><td>The alphabetical index at the back: terms and the pages they appear on.</td><td>never</td>
<td>The concept list for the skeleton; Discover reads it last; Library search with <i>book index regions</i> lists its
entries and, once you set <i>Printed page number</i> on a page, turns the printed numbers it cites into scan pages.</td></tr>
<tr><td><b>table of cases</b>, <b>table of statutes</b></td><td>A law book's lists of the cases and Acts it cites.</td>
<td>never</td><td>Added to the skeleton with the book index (which authorities the book relies on). Not read by Discover.</td></tr></table>
<p>A table of contents, book index or table is always given the role <b>context</b>: the pages describe the book, they
do not teach, so training on them would teach page numbers.</p>
<h4>Roles</h4>
<table class="tbl"><tr><th>Role</th><th>Meaning</th></tr>
<tr><td><b>train</b></td><td>Learned from: read by Discover and mined for the playbook, facts, examples and training items.</td></tr>
<tr><td><b>context</b></td><td>For table of contents, book index and tables: structure only. On a chapter it does almost
nothing (the chapter is named to Discover but never read or mined).</td></tr>
<tr><td><b>heldout</b></td><td>Never trained on. Used only for the honest final tests: the back-test on held-out regions
and the transfer pilot. Tag at least one chapter this way, with problems the training chapters do not contain.</td></tr>
<tr><td><b>exclude</b></td><td>Ignored everywhere (covers, prefaces, adverts, pages you do not want read).</td></tr></table>
<h4>Then add them to a topic</h4>
<p>A region is used only by the topics it belongs to: tick it in <b>Topics</b> and save, or tick <i>every region of
these documents, now and later</i>. The topic shows how many train regions and pages will be mined, and says why when
there are none. Type and role can be changed in the region list here; the region keeps its id, so topics keep it.
Deleting a region and tagging it again gives it a new id, and topics lose it.</p>` },
  { id: "topics", tab: "topics", level: "basic", title: "Topics", html: `
<p>A topic gathers regions from any documents, plus curated data sets, into what will become one pack. Give it a clear
name and say what it is about: generation uses that text. Curated data (Easy Dataset exports, Obsidian notes and
dictionaries, spreadsheets, JSON/YAML/XML, HTML …) is added at the bottom.</p>` },
  { id: "generate", tab: "generate", level: "basic", title: "Generate: building a dataset", html: `
<p>Each press of <b>Generate a new dataset</b> creates a dataset you can keep, compare or delete.</p>
<ul><li><b>Facts &amp; data only</b>: captures facts, tables, entity profiles, procedures, recommendations and terms page
by page; every number must be printed on its page. Scales with pages.</li>
<li><b>With a playbook</b>: worked cases, minimal pairs, next-check cases and scenes per check — or a built-in playbook.</li>
<li><b>Both</b>: facts and reasoning together; reasoning is repeated to reach its share.</li></ul>
<p><b>Pause</b> stops at once and keeps finished work; <b>Continue</b> (in the list below, or Datasets) does only what is
missing or failed. Every model call is cached, so nothing is paid for twice.</p>` },
  { id: "mining-basic", tab: "generate", level: "basic", title: "Extra mining: knowing vs performing", html: `
<p>The facts track makes a pack <b>know</b>; the playbook makes it <b>reason</b>. Neither teaches it to <b>produce the
expert's work</b> — a draft like the book's forms, a report like its specimens, prose in an author's voice — or to
<b>review</b> someone's work, or to know <b>where its knowledge stops</b>. The <b>Extra mining</b> section (under
“Dataset from”, folded away and empty by default) adds that.</p>
<ul><li>Leave it empty: nothing changes.</li>
<li>Tick miners (or pick a profile): they are added to whatever “Dataset from” makes.</li>
<li>“Dataset from: <b>Only the extra mining I tick</b>”: nothing else is mined — e.g. only exemplars, or only the full text.</li></ul>
<p>Profiles: <b>Writer</b> (write in an author's voice), <b>Drafter</b> (documents like the book's forms and clauses),
<b>Reader / reporter</b> (from findings or facts to the expert's report), <b>Reviewer</b> (check and correct work).</p>` },
  { id: "mining-each", tab: "generate", level: "intermediate", title: "What each miner does", html: `
<table class="tbl"><tr><th>Miner</th><th>Takes from the book</th><th>Teaches</th></tr>
<tr><td>Exemplars</td><td>finished outputs word for word (clauses, forms, specimen reports, letters, passages); anything not found word for word on the page is dropped</td><td>perform</td></tr>
<tr><td>Variants</td><td>alternatives with the condition for each</td><td>choosing, not copying</td></tr>
<tr><td>Corrections</td><td>errors → fixes, before → after</td><td>review</td></tr>
<tr><td>Checklists</td><td>review lists, search patterns, protocols</td><td>working in order</td></tr>
<tr><td>Structures</td><td>how each document or report is laid out</td><td>long documents in the right order</td></tr>
<tr><td>Limits</td><td>scope, assumptions, jurisdiction or era, when to refer, what may be outdated</td><td>knowing where to stop</td></tr>
<tr><td>Cases</td><td>input → the expert's output (some kept as tests)</td><td>perform on a case</td></tr>
<tr><td>Requests → text</td><td>for each exemplar or passage, the request that would produce it (written by the LLM); trained request → the book's own text</td><td>perform on request</td></tr>
<tr><td>Plain → styled</td><td>a flat version of each passage; trained flat → original</td><td>the voice</td></tr>
<tr><td>Flawed → fixed</td><td>a flawed version of each exemplar with its defects; trained to name them and give the correct text</td><td>review</td></tr>
<tr><td>Full text</td><td>the whole text in overlapping windows, nothing dropped</td><td>voice and flow</td></tr>
<tr><td>Voice profile</td><td>sentence length, short/long sentences, “and”-chains, dialogue, word length (shown in the report)</td><td>measuring style</td></tr></table>` },
  { id: "mining-advanced", tab: "generate", level: "advanced", title: "Performing packs: training, limits, safety", html: `
<ul><li><b>Training</b>: performing changes most words the model writes, so Train suggests large capacity, KL anchor 0.3
and Q&amp;A in chat format for datasets with performing miners (untested on the 30B model — compare with the base).</li>
<li><b>Long outputs</b> (a 4,000-word story, a full agreement) are written in parts — outline → sections — not in one answer.</li>
<li><b>Legal drafting</b>: a pack is a first-draft assistant within the book's jurisdiction and era, under a lawyer's
review. Tick <i>Limits</i> to learn the book's assumptions; check statutory references against the fact book (packs cannot
cite sources); a book written before 2023 teaches pre-2023 statutes.</li>
<li><b>Images</b>: a text model cannot see an X-ray. A radiology pack can learn the search order, differentials and
report writing from described findings — education and second reading, not diagnosis.</li>
<li><b>Copying</b>: with one source text and the full-text miner, a pack may repeat its sentences; compare outputs with the book.</li></ul>` },
  { id: "traces", tab: "generate", level: "advanced", title: "Reasoning traces: teaching the process, verified", html: `
<p>A <b>trace</b> spells out how a problem gets to its answer as ordered <b>anchors</b>, each with its result, the words it
rests on (from the problem or the reference) and what it depends on. Legal: facts → issue → law → precedent →
interpretation → application → counterargument → conclusion. Medical: presentation → findings → pattern → differential →
discriminators → working diagnosis → red flags → management. General: givens → question → rules → working → check →
exceptions → result.</p>
<ol><li><b>Sources</b>: the text's worked examples (as the back-test finds them), the Cases miner, curated Q&amp;A.</li>
<li><b>Solve first</b>, without the answer, using the playbook (or the fact book) as reference; kept when the conclusion matches the source.</li>
<li>Otherwise <b>reconstruct</b> with the answer, then <b>re-check</b>: the steps alone, conclusion removed, must lead to the source's conclusion.</li>
<li><b>Checks</b>: evidence must be in the problem or reference; provisions and cases must be in the sources (otherwise dropped);
old codes (IPC, CrPC, Indian Evidence Act) flagged; anchors must depend on earlier ones; the counterargument must be specific.</li></ol>
<p>Only traces that pass become training items; every trace and its checks are kept in <code>topics/&lt;id&gt;/mining/traces.jsonl</code>.
Train such datasets with <b>answer-only loss</b> and the <b>curriculum</b> (Train suggests them).</p>
<p><b>Literature</b>: there is no answer to check a craft decision against, so craft traces are unverified rationales —
made only when you tick <i>craft notes (unverified)</i>. For an author's voice use the Writer miners.</p>
<p><b>Is it working?</b> Run the <b>transfer pilot</b> (Workbench) before generating thousands of traces.</p>` },
  { id: "pilot", tab: "workbench", level: "advanced", title: "Transfer pilot", html: `
<p>Tests whether reasoning <b>transfers</b> or the pack only learns the format. From the topic's problems with known
answers it builds two datasets over exactly the same kept problems — <b>A</b>: question → answer, <b>B</b>: question →
verified anchored trace → answer — trains both packs with the same settings, and tests both on a <b>held-out chapter</b>
(tag one as <i>heldout</i> in the Library: new problem families, not rewordings).</p>
<p>The report compares held-out conclusion scores (base, A, B) and <b>reasoning discipline</b> on the new problems: how
often each reasons in anchors, invents provisions or cases, quotes evidence that is not in the problem, or cites old codes.
Start with about 100 problems; if B beats A on the held-out chapter, scale up; if it only looks tidier, revise before generating more.</p>` },
  { id: "loss-curriculum", tab: "train", level: "advanced", title: "Answer-only loss and curriculum", html: `
<ul><li><b>Learn: only the reasoning and answer</b> — for Q&amp;A and trace items the question is context and is not learned
(the model is not trained to write questions). Other items (facts, passages) are learned whole.</li>
<li><b>Curriculum</b> — three stages from easy to hard (by kind: facts → cases → traces and reviews; and by length), each
for the chosen epochs; the last stage replays 10% of the easiest items so short answers do not degrade. The best epoch is
taken from the last stage.</li></ul>` },
  { id: "datasets", tab: "datasets", level: "basic", title: "Datasets: look, search, edit", html: `
<p>Every dataset (generated, composed, or imported from curated data) is listed with its status and counts.
Open one to see its items.</p>
<ul><li><b>Search</b> words or an exact phrase; tick <i>pattern</i> for a regular expression (e.g. <code>Section \\d+</code>).
Filter by kind, source, edited/reviewed, or items with numbers.</li>
<li><b>Select all matches</b>, then delete, change kind, mark reviewed, or move between training and tests.</li>
<li><b>Find / replace</b> on the selection: Preview shows every change first; Apply makes it.</li>
<li>Click an item's text to edit it; <b>+ Add item</b> writes a new one.</li>
<li>Every change is a <b>version</b>: revert to any earlier version. Edits are kept when you Continue a generation.</li></ul>` },
  { id: "composer", tab: "composer", level: "basic", title: "Composer: combining datasets", html: `
<p>Tick any datasets — reasoning, facts-only, curated imports, earlier compositions. The preview shows the combined count
after duplicates are removed. When two or more come with playbooks, cross-topic cases are written (situations that need
both). Packs for the same domain should be trained together from a composition rather than merged afterwards.</p>` },
  { id: "train", tab: "train", level: "basic", title: "Train: settings that matter", html: `
<ul><li><b>Epochs</b>: passes over every item (each item seen once per epoch, reshuffled). <b>auto</b> is the default: about 4,800 texts in all, 2 to 16 epochs.</li>
<li><b>Training length</b> 1024 tokens with <b>batch 2 × accumulation 4</b> works on a 32 GB card; batch 8 runs out of memory.</li>
<li><b>Capacity</b>: standard (2 experts × 4 layers) suits up to a few thousand items; large (4 × 6) for 6,000+.</li>
<li>After each epoch the loss on the dataset's tests is measured and the <b>best epoch is kept</b>.</li>
<li><b>Stop and save</b> ends after the current step and exports the best weights so far; <b>Cancel</b> discards.</li></ul>` },
  { id: "workbench", tab: "workbench", level: "basic", title: "Workbench: asking and evaluating", html: `
<ul><li><b>Chat format</b> uses the base model's chat template, a system prompt and the conversation so far — longer,
more coherent answers than the Question/Answer format. Temperature 0.2–0.4 avoids loops; raise max tokens for detail.
The engine's tokens per request (Settings) caps prompt + answer.</li>
<li><b>Compare</b> answers the same question with the base and with each ticked pack.</li>
<li><b>Analyse</b> on an answer checks that answer against what its pack was built from: a user pack's statements and text, or the pack's own topic (fact book, data, playbook, pages); the topic at the top only when the pack has none. Claims are <i>right</i>, <i>wrong</i> (contradicted by the material), <i>not covered</i> (the material does not say: not an error) or <i>missing</i>. The result names what it checked against.</li>
<li><b>Evaluate</b> runs a dataset's tests with and without the pack; results open in a table (filter: pack wrong,
worse than base) and stay under Earlier evaluations and in Jobs.</li></ul>` },
  { id: "assistant", tab: "assistant", level: "basic", title: "Assistant", html: `
<p>Tell it what you want learned and from which document; attach documents from the library or upload one. It asks a
few questions, shows a plan card, and on <b>Run</b> builds the dataset — and trains and tests if you want. Pause, resume
or cancel from the chat or from Jobs. Training unloads the LLM server's models (the assistant too) and loads them again after.</p>` },
  { id: "jobs", tab: "jobs", level: "basic", title: "Jobs", html: `
<p>Every long task is a job, kept on disk: open one for its full log and results (evaluation tables, dataset reports,
training epochs). <b>Pause</b> stops at once and keeps finished work; <b>Resume</b> runs it again and skips what is done.
Delete finished jobs individually or all at once.</p>` },
  { id: "settings", tab: "settings", level: "basic", title: "Settings", html: `
<p>Settings are grouped; each has a short explanation. The <b>Parallel</b> box in the header is the same setting.
Settings marked “restart” take effect after restarting Packsmith (or the engine).</p>` },

  // ---------------------------------------------------------------- reference
  { id: "best", tab: "*", level: "intermediate", title: "Best settings (tested)", html: `
<table class="tbl"><tr><th>Setting</th><th>Value</th><th>Why</th></tr>
<tr><td>Mining model</td><td>Gemma 4 31B (or similar), context 32K–128K</td><td>tight plans, complete capture</td></tr>
<tr><td>Text per call</td><td>32,000 tokens</td><td>mapping sees whole chapters; the reply needs room on top</td></tr>
<tr><td>Text per capture call</td><td>8,000 tokens</td><td>capture must write out all it reads</td></tr>
<tr><td>Reply budget</td><td>16,384 (more for thinking models)</td><td>thinking counts inside the reply</td></tr>
<tr><td>Parallel</td><td>2–4, ≤ the server's parallel slots</td><td>more only queues</td></tr>
<tr><td>Training length</td><td>1024</td><td>2048+ untested on the 30B model</td></tr>
<tr><td>Batch × accumulation</td><td>2 × 4</td><td>8 runs out of memory at 1024 on 32 GB</td></tr>
<tr><td>Epochs</td><td>2</td><td>3+ memorises wordings and generalises less</td></tr>
<tr><td>Capacity</td><td>standard; large for 6,000+ items</td><td>capacity, not epochs, limits large sets</td></tr>
<tr><td>Workbench</td><td>chat format, temperature 0.3, 1024+ tokens</td><td>long, non-looping answers</td></tr>
<tr><td>One shared GPU</td><td>on (single 32 GB card)</td><td>engine and the LLM server take turns: no VRAM overload</td></tr></table>
<p>Data design matters more than settings (from the experiments): facts stated inside realistic scenes help a lot;
many wordings per fact help; state facts positively; reasoning needs the facts present; packs for the same domain must be trained together.</p>` },
  { id: "sharedgpu", tab: "settings", level: "intermediate", title: "One GPU for the engine and the LLM server", html: `
<p>With <b>Where the LLM server runs = same card</b> on (Settings → GPU), only one of them holds VRAM at a time,
so the 30B engine and a 31B mining model never overload the card together:</p>
<ul><li>any LLM work (mining, generation, the assistant) stops the engine first and loads again the LM Studio
models Packsmith had unloaded, with the context they had;</li>
<li>asking in the Workbench (or Start) unloads the LLM server's models and starts the engine;</li>
<li>judging answers, <i>Analyse</i>, and an evaluation's judging phase do: <b>stop engine → judge → unload the LLM →
start the engine again</b>;</li>
<li>training takes the whole GPU, then restarts the engine if it was running.</li></ul>
<p>The header badge shows who holds the GPU. Starting the engine takes as long as loading the base model, so the first
Workbench question after mining waits for it. Turn the setting off when you have two GPUs or enough VRAM for both.</p>` },
  { id: "testtab", tab: "test", level: "basic", title: "The Test tab", html: `
<p>The same questions to several <b>candidates</b> (an engine with packs at set strengths and routing), answers side by
side. The values on a row are for the test only: never a pack's saved dials, never what the Workbench has loaded.</p>
<ul><li><b>Rows</b>: one per installed engine at first; <i>Add test</i> adds one; ⧉ duplicates (10%, 100%, 150% is three
rows). The base model on each engine is added as a reference. A pack that does not fit shows why and cannot be ticked;
routing is for Polymath route packs only.</li>
<li><b>Questions</b>: blocks sent in order. <i>Keep the conversation</i> off: each question on its own; on: one
conversation per candidate, each continuing from its own answers. An expected answer marks right or wrong by a
whole-word match.</li>
<li><b>Run</b>: a background job. The Workbench's engine is stopped for it and put back afterwards as it was, also when
the run is cancelled. Each engine is started once; each answer is saved as it arrives, so a paused run continues where
it stopped.</li>
<li><b>Results</b>: speed, cut off, routing share, think never closed, repeating, same as base and matches expected,
without a judge. <i>Evaluate</i> uses the Analyse check, blind: the judge is not told which candidate wrote an
answer.</li></ul>` },
  { id: "lab", tab: "lab", level: "advanced", title: "The Train Lab", html: `
<p>Ways of training a pack tried on the same data and compared by what the packs answer. The Lab only trains and
scores: the Train tab, its defaults and the library are not changed, and a pack trained from the Train tab is the same,
bit for bit, as before the Lab existed. Lab packs stay in the Lab's folder.</p>
<ul><li><b>Baseline</b>: today's Train tab defaults, always trained first. Each ready-made experiment differs from it in
one respect; a recipe that differs in more than one is marked, since the cause of a change cannot then be told apart.</li>
<li><b>Dataset</b>: the yardstick (the invented harbour set: 450 items, 1,080 questions, answers the base model cannot
know) or any dataset from the Datasets tab (its tests, matched by keywords; 40 of them are kept apart for the quick
score after each epoch and never counted in the final table).</li>
<li><b>Scores</b>: every cell is a share with its count and 95% interval, e.g. 67% (101 of 150, 59 to 74). A cell is
coloured better or worse than the baseline only when the two intervals do not overlap. On chat models each recipe is
scored with thinking on and with thinking off, as separate tables, with "thinking never closed", "thinking skipped
although on" and "repeating".</li>
<li><b>Thinking options</b> (off by default): <i>trained both ways</i> trains a share of the Q&amp;A items also the
other way (filled thinking block as well as empty; the filling is the item's own reasoning or quoted evidence, never
invented). <i>loss on</i>: all tokens; answer (reasoning and answer); final (only the final answer).</li>
<li><b>Per epoch</b>: held-out loss and the quick sample after each epoch; with <i>score every epoch</i> a copy of the
pack after each epoch is kept and scored in full, so you can see whether the kept epoch was the best one.</li>
<li><b>Reasoning dataset</b> (made only when you ask): the chosen dataset's items unchanged, plus reasoning items as
<i>Question / Reasoning / Answer</i> from the text's worked problems, its argument passages (each step quoting the
passage), linked facts joined into 2- and 3-step questions, and, only if ticked, rules applied to invented situations
(marked constructed). Every item is checked (quotations word for word, provisions and cases in the source, the steps
alone re-solved to the same conclusion); dropped items are listed with the reason. Tests are split off first. Long items
are never cut: split at step boundaries, left out and counted, or kept with the training length raised.
<i>Generate reasoning blocks</i> adds each reasoning item with its reasoning in the thinking block (Qwen);
<i>extract reasoning only</i> keeps only the reasoning and the items it rests on. Nothing is written into the topic.</li>
<li><b>Items trained</b> (recipes "without the reasoning" and "reasoning with its facts only"): the same reasoning
dataset and the same tests, a different share of the items, so what the reasoning adds shows in one table.</li>
<li><b>Test packs</b>: <i>Add to library (test_lab)</i> under a recipe's results. Tick it by hand in the Workbench or
the Test tab; the assistant and agents never choose it. <i>Promote</i> makes it a normal pack; <i>Delete</i> is final.</li></ul>` },
  { id: "storage", tab: "storage", level: "basic", title: "The Storage tab", html: `
<p>Everything Polymath has put on this machine for you, wherever it is: the data folder, the packs and models folders,
the Hugging Face download cache, the runs folder, Polymath's own folder, and leftovers in the temporary folder. One base
model can take space three times (the Hugging Face download, Polymath's models folder, the colibri container), so
the tab shows them all. Choose a category, tick what to delete, then Delete.</p>
<ul><li><b>Datasets</b>, <b>Packs</b>, <b>Models</b> (colibri containers, model folders, Hugging Face downloads; a
download already converted for colibri is marked), <b>Training checkpoints and runs</b> (checkpoints, Train Lab runs and
epoch copies, regression outputs), <b>Caches and temporary files</b> (merged packs, LLM-call caches, Test runs, logs,
Hugging Face's and pip's download caches, files left by stopped engines and test runs).</li>
<li>Deleting is <b>final</b>. More than 1 GB or more than 5 items needs DELETE typed.</li>
<li>Items in use cannot be ticked: the base model in Settings, a pack loaded in the Workbench, a running job's run or
dataset, an engine still running.</li>
<li>Only those kinds, in their own folders; a link is never followed out. With the API token on it is required.</li>
<li>Never listed: documents, topics, playbooks, settings, jobs and the engines.</li></ul>` },
  { id: "epochs", tab: "train", level: "advanced", title: "How much training a pack needs", html: `
<p><b>auto</b> (the default) gives about 4,800 texts seen in all: 11 epochs for 450 items, 2 from 2,400 items, at most
16. That is the budget the moe-knowledge packs that worked had (300 to 1,200 steps of 8).</p>
<ul><li>Until 2026-11-13 the default was 2 epochs. In the regression test (<code>scripts/regression</code>) a 450-item
pack trained that way learned <b>nothing</b>: 0% on its questions, the same as the base. At 10 epochs it answered 99%
of its facts in new wordings and 90% of unseen two- and three-hop questions.</li>
<li>The trainer keeps the epoch with the lowest loss on the held-out tests, so extra passes that start to overfit do
not make the pack worse.</li>
<li>A pack's experts now start as copies of the experts its text uses most (silent at first), and the KL anchor also
holds the pack to the base's own answers to ordinary questions. Both are automatic.</li></ul>` },
  { id: "trouble", tab: "*", level: "advanced", title: "Troubleshooting", html: `
<ul><li><b>“Spent the whole budget thinking” / empty replies</b>: the server counts a thinking model's reasoning inside the
reply budget (max_tokens). Raise Settings → <i>Reply budget</i> (default 16,384; every call gets at least this), or set
<i>Reasoning effort</i> to low. Packsmith also retries once with twice the room when a reply ends while still thinking.
<i>Text per call</i> is the input budget, not the reply.</li>
<li><b>Mapping gives 0 fragments / empty replies</b>: the model ran out of reply budget thinking, or the prompt did
not fit. Turn thinking off, load with more context; look in <code>fragments/raw/</code>.</li>
<li><b>Training runs out of memory</b>: lower batch to 1–2 and raise accumulation; shorten the training length.</li>
<li><b>LLM server not unloading</b>: only LM Studio (<code>/api/v1/models/unload</code>) and Ollama (<code>keep_alive: 0</code>) can be unloaded; Packsmith calls them
per loaded instance; check the server is on and the API key (if any) is set.</li>
<li><b>Short or looping Workbench answers</b>: use chat format, temperature 0.3, more max tokens.</li>
<li><b>A run failed half way</b>: open it in Jobs and Resume, or Continue the dataset: finished work is not repeated.</li></ul>` },
  { id: "glossary", tab: "*", level: "basic", title: "Glossary", html: `
<dl><dt>Pack</dt><dd>extra expert layers trained for one base model; switched on or off at run time.</dd>
<dt>Move</dt><dd>a step experts take (check an element, distinguish a look-alike…).</dd>
<dt>Mining plan</dt><dd>the moves this text makes and what to extract for each.</dd>
<dt>Fragment</dt><dd>one place where a page makes a move, with its parts and page.</dd>
<dt>Playbook</dt><dd>the ordered checks built from all fragments.</dd>
<dt>Check</dt><dd>one step of the playbook, with branches, cues, exceptions, distinctions, facts.</dd>
<dt>Back-test</dt><dd>solving the book's own examples with only the playbook.</dd>
<dt>Dataset</dt><dd>training items and tests made by Generate, Composer or an import.</dd>
<dt>Epoch</dt><dd>one pass over every training item.</dd>
<dt>Held-out</dt><dd>kept out of training, used to measure.</dd></dl>` },
];
const LEVELS = { basic: "Basics", intermediate: "Using it well", advanced: "Advanced" };

function helpFor(tab) { return HELP.filter((h) => h.tab === tab || (tab === "*" && h.tab === "*")); }
function helpHTML(list) {
  return ["basic", "intermediate", "advanced"].map((lv) => {
    const xs = list.filter((h) => h.level === lv);
    return xs.length ? `<h3 class="lv">${LEVELS[lv]}</h3>` + xs.map((h) => `<section id="h-${h.id}"><h4>${h.title}</h4>${h.html}</section>`).join("") : "";
  }).join("");
}
function openHelpDrawer(tab) {
  const list = helpFor(tab).length ? helpFor(tab) : helpFor("*");
  const name = (document.querySelector(`#tabs button[data-tab="${tab}"]`) || {}).textContent || "Help";
  document.getElementById("helpDrawerTitle").textContent = "Help · " + name;
  document.getElementById("helpDrawerBody").innerHTML = helpHTML(list);
  document.getElementById("helpDrawer").classList.remove("hidden");
  document.getElementById("helpDrawerFull").onclick = () => { document.getElementById("helpDrawer").classList.add("hidden");
    document.querySelector('#tabs button[data-tab="help"]').click(); showHelpTopic((list[0] || {}).id); };
}
function renderHelpTab() {
  const groups = [["*", "Getting started & reference"], ["assistant", "Assistant"], ["library", "Library"], ["topics", "Topics"],
    ["approaches", "Approaches"], ["discover", "Discover"], ["playbook", "Playbook"], ["generate", "Generate"], ["datasets", "Datasets"],
    ["composer", "Composer"], ["train", "Train"], ["workbench", "Workbench"], ["jobs", "Jobs"], ["settings", "Settings"]];
  document.getElementById("helpToc").innerHTML = groups.map(([tab, name]) => {
    const xs = HELP.filter((h) => h.tab === tab);
    return xs.length ? `<div class="small"><b>${name}</b></div>` + xs.map((h) => `<a href="#" class="helplink" data-h="${h.id}">${h.title} <span class="muted">· ${LEVELS[h.level]}</span></a>`).join("") : "";
  }).join("");
  document.querySelectorAll(".helplink").forEach((a) => a.onclick = (e) => { e.preventDefault(); showHelpTopic(a.dataset.h); });
  document.getElementById("helpDoc").innerHTML = groups.map(([tab, name]) => {
    const xs = HELP.filter((h) => h.tab === tab);
    return xs.length ? `<h2>${name}</h2>` + helpHTML(xs) : "";
  }).join("");
}
function showHelpTopic(id) { const el = document.querySelector(`#helpDoc #h-${id}`); el && el.scrollIntoView({ behavior: "smooth" }); }

// a "?" button on every tab
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("section.tab").forEach((s) => {
    const tab = s.id.replace("tab-", "");
    if (tab === "help") return;
    const b = document.createElement("button");
    b.className = "helpbtn"; b.textContent = "? Help"; b.title = "How to use this tab, from the basics to advanced use";
    b.onclick = () => openHelpDrawer(tab);
    s.prepend(b);
  });
  document.getElementById("helpDrawerClose").onclick = () => document.getElementById("helpDrawer").classList.add("hidden");
  renderHelpTab();
});

// ------------------------------------------------------------------ web view editors
// The web view is a contenteditable rendering of the stored HTML. The browser parses (and so repairs) the markup; edits
// are written back to the textarea on every change, so Save always sends well-formed HTML. The text view is the raw
// HTML; switching back re-parses it.
function cleanHTML(el) {
  const c = el.cloneNode(true);
  c.querySelectorAll("[contenteditable]").forEach((x) => x.removeAttribute("contenteditable"));
  c.querySelectorAll(".edsel").forEach((x) => x.classList.remove("edsel"));
  c.querySelectorAll(".edempty").forEach((x) => x.remove());
  c.querySelectorAll("[class='']").forEach((x) => x.removeAttribute("class"));
  return c.innerHTML.replace(/\u200b/g, "").trim();
}
function makeHtmlEditor(cfg) {
  // cfg: {text, view, tools, tabs, empty, kinds: [...], blocks: {label: html}, itemSel, blockSel}
  const ta = document.getElementById(cfg.text), view = document.getElementById(cfg.view), tools = document.getElementById(cfg.tools);
  let mode = "web", sel = null;
  const sync = () => { if (mode === "web") ta.value = cleanHTML(view); };
  const load = () => { view.innerHTML = safeHTMLlite(ta.value) || `<p class="muted edempty" contenteditable="false">${cfg.empty}</p>`; };
  view.addEventListener("input", () => { if (mode === "web") ta.value = cleanHTML(view); });
  view.addEventListener("click", (e) => {
    const it = e.target.closest(cfg.itemSel) || e.target.closest(cfg.blockSel);
    if (sel) sel.classList.remove("edsel");
    sel = it && view.contains(it) ? it : null;
    if (sel) sel.classList.add("edsel");
    if (kindSel) kindSel.value = sel ? (cfg.kinds.find((k) => sel.classList.contains(k)) || "") : "";
  });
  let kindSel = null;
  if (tools) {
    tools.innerHTML = Object.keys(cfg.blocks).map((k) => `<button data-add="${k}" class="ghost">+ ${k}</button>`).join("") +
      `<select class="edkind"><option value="">kind of line…</option>${cfg.kinds.map((k) => `<option>${k}</option>`).join("")}</select>` +
      `<button data-act="up" class="ghost" title="move up">↑</button><button data-act="down" class="ghost" title="move down">↓</button>` +
      `<button data-act="del" class="ghost">Delete selected</button><button data-act="undo" class="ghost">Undo</button><button data-act="redo" class="ghost">Redo</button>` +
      `<span class="muted small">click a line to select it</span>`;
    kindSel = tools.querySelector(".edkind");
    kindSel.onchange = () => { if (sel && kindSel.value) { cfg.kinds.forEach((k) => sel.classList.remove(k)); sel.classList.add(kindSel.value); ta.value = cleanHTML(view); } };
    tools.querySelectorAll("[data-add]").forEach((b) => b.onclick = () => {
      const tpl = document.createElement("template"); tpl.innerHTML = cfg.blocks[b.dataset.add].trim();
      const node = tpl.content.firstElementChild;
      const anchorSel = node.matches(cfg.blockSel) ? cfg.blockSel : cfg.itemSel;
      const anchor = sel && (sel.matches(anchorSel) ? sel : sel.closest(anchorSel));
      if (!view.querySelector(cfg.blockContainer)) { view.querySelectorAll(".edempty").forEach((x) => x.remove()); view.insertAdjacentHTML("beforeend", cfg.skeleton); }
      if (anchor) anchor.after(node);
      else if (node.matches(cfg.blockSel)) view.querySelector(cfg.blockContainer).appendChild(node);
      else { const host = sel ? (sel.querySelector("ul,ol") || sel) : (view.querySelector(cfg.itemContainer) || view); host.appendChild(node); }
      ta.value = cleanHTML(view); node.scrollIntoView({ block: "nearest" });
    });
    tools.querySelectorAll("[data-act]").forEach((b) => b.onclick = () => {
      const a = b.dataset.act;
      if (a === "undo" || a === "redo") { view.focus(); document.execCommand(a); ta.value = cleanHTML(view); return; }
      if (!sel) return toast("click a line or a card first");
      if (a === "del") { const nx = sel.nextElementSibling || sel.previousElementSibling; sel.remove(); sel = null; if (nx) { nx.classList.add("edsel"); sel = nx; } }
      if (a === "up" && sel.previousElementSibling) sel.previousElementSibling.before(sel);
      if (a === "down" && sel.nextElementSibling) sel.nextElementSibling.after(sel);
      ta.value = cleanHTML(view);
    });
  }
  document.querySelectorAll(`.viewtabs[data-for="${cfg.text}"] button`).forEach((b) => b.onclick = () => {
    const to = b.dataset.v;
    if (to === mode) return;
    if (to === "web") { load(); } else { sync(); }
    mode = to;
    document.querySelectorAll(`.viewtabs[data-for="${cfg.text}"] button`).forEach((x) => x.classList.toggle("on", x.dataset.v === to));
    ta.classList.toggle("hidden", to !== "text"); view.classList.toggle("hidden", to !== "web"); tools && tools.classList.toggle("hidden", to !== "web");
  });
  return { load: () => { load(); }, sync };
}
function safeHTMLlite(h) { return String(h || "").replace(/<script[\s\S]*?<\/script>/gi, "").replace(/\son\w+="[^"]*"/gi, ""); }

const PLAN_KINDS = ["desc", "extract", "example", "about", "confidence"];
const PB_KINDS = ["branch", "cue", "exception", "distinction", "fact", "step", "source", "note"];
let PlanEditor, PbEditor;
document.addEventListener("DOMContentLoaded", () => {
  PlanEditor = makeHtmlEditor({ text: "dPlan", view: "dPlanView", tools: "dPlanTools", empty: "No plan yet: run Discover, or load the latest draft.",
    kinds: PLAN_KINDS, itemSel: "p, blockquote", blockSel: "section.move", blockContainer: "section.plan", itemContainer: "section.move",
    skeleton: `<section class="plan"><p class="about">What this text is about.</p></section>`,
    blocks: { move: `<section class="move" data-name="new-move" data-pattern="element"><p class="desc">What the author does here.</p><p class="extract">what to pull out of each page</p></section>`,
      "extract line": `<p class="extract">what to pull out</p>`, example: `<blockquote class="example"><cite data-doc="" data-page=""></cite>example</blockquote>` } });
  PbEditor = makeHtmlEditor({ text: "pEdit", view: "pView", tools: "pTools", empty: "No playbook yet.",
    kinds: PB_KINDS, itemSel: "li:not(.check)", blockSel: "li.check", blockContainer: "ol.checks", itemContainer: "li.check ul",
    skeleton: `<section class="playbook"><ol class="checks"></ol></section>`,
    blocks: { check: `<li class="check" data-id="cX" data-move=""><span class="label">New check</span><ul><li class="step">a step</li></ul></li>`,
      line: `<li class="fact">a fact this check relies on</li>` } });
  window.PlanEditor = PlanEditor; window.PbEditor = PbEditor;
});

// ------------------------------------------------------------------ approach record: web view form
const RecEditor = {
  mode: "web",
  load() {
    let r = {};
    try { r = JSON.parse(document.getElementById("recEdit").value || "{}"); } catch (e) { r = null; }
    const host = document.getElementById("recWeb");
    if (!r) { host.innerHTML = `<p class="small" style="color:var(--bad)">The JSON has an error: fix it in Text view.</p>`; return; }
    const pats = (S.lib && S.lib.patterns || []).map((p) => p.name);
    const mv = (m, i) => `<div class="mvcard" data-i="${i}">
      <div class="row"><input class="mv-name" value="${esc(m.name || "")}" placeholder="move name">
        <select class="mv-pat">${["", ...pats].map((p) => `<option ${p === m.pattern ? "selected" : ""}>${esc(p)}</option>`).join("")}</select>
        <button class="ghost mv-up">↑</button><button class="ghost mv-del">✕</button></div>
      <textarea class="mv-desc" rows="2" placeholder="what the expert does">${esc(m.desc || "")}</textarea>
      <textarea class="mv-ex" rows="2" placeholder="what to extract">${esc(m.extract || "")}</textarea></div>`;
    host.innerHTML = `<div class="grid2"><label>Domain <input id="rf-domain" value="${esc(r.domain || "")}"></label>
      <label>Intent <select id="rf-intent">${["reasoning", "craft", "diagnostic", "procedure", "mixed"].map((x) => `<option ${x === r.intent ? "selected" : ""}>${x}</option>`).join("")}</select></label></div>
      <label>Id <input id="rf-id" value="${esc(r.id || "")}"></label>
      <h4>Moves</h4><div id="rf-moves">${(r.moves || []).map(mv).join("")}</div><button id="rf-add" class="ghost">+ move</button>
      <h4>Situations (one per line)</h4><textarea id="rf-sit" rows="4">${esc((r.situations || []).join("\n"))}</textarea>
      ${r.scores ? `<h4>Scores</h4><div class="small muted">${esc(JSON.stringify(r.scores))}</div>` : ""}`;
    host.oninput = () => RecEditor.sync();
    host.querySelector("#rf-add").onclick = () => { host.querySelector("#rf-moves").insertAdjacentHTML("beforeend", mv({}, Date.now())); RecEditor.wire(); RecEditor.sync(); };
    RecEditor.wire();
  },
  wire() {
    document.querySelectorAll("#rf-moves .mv-del").forEach((b) => b.onclick = () => { b.closest(".mvcard").remove(); RecEditor.sync(); });
    document.querySelectorAll("#rf-moves .mv-up").forEach((b) => b.onclick = () => { const c = b.closest(".mvcard"); c.previousElementSibling && c.previousElementSibling.before(c); RecEditor.sync(); });
    document.querySelectorAll("#rf-moves select").forEach((s) => s.onchange = () => RecEditor.sync());
  },
  sync() {
    if (RecEditor.mode !== "web" || !document.getElementById("rf-domain")) return;
    let r = {};
    try { r = JSON.parse(document.getElementById("recEdit").value || "{}"); } catch (e) { r = {}; }
    r.domain = document.getElementById("rf-domain").value; r.intent = document.getElementById("rf-intent").value;
    if (document.getElementById("rf-id").value) r.id = document.getElementById("rf-id").value;
    r.moves = [...document.querySelectorAll("#rf-moves .mvcard")].map((c) => ({ name: c.querySelector(".mv-name").value.trim(),
      pattern: c.querySelector(".mv-pat").value, desc: c.querySelector(".mv-desc").value.trim(), extract: c.querySelector(".mv-ex").value.trim() }))
      .filter((m) => m.name);
    r.situations = document.getElementById("rf-sit").value.split("\n").map((s) => s.trim()).filter(Boolean);
    document.getElementById("recEdit").value = JSON.stringify(r, null, 1);
  },
};
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll('.viewtabs[data-for="rec"] button').forEach((b) => b.onclick = () => {
    const to = b.dataset.v;
    if (to === RecEditor.mode) return;
    if (to === "text") RecEditor.sync();
    RecEditor.mode = to;
    if (to === "web") RecEditor.load();
    document.querySelectorAll('.viewtabs[data-for="rec"] button').forEach((x) => x.classList.toggle("on", x.dataset.v === to));
    document.getElementById("recEdit").classList.toggle("hidden", to !== "text");
    document.getElementById("recWeb").classList.toggle("hidden", to !== "web");
  });
});
