</head>

<body class="fullcontent">


  


</header>


<div id="00a794db-caa0-4053-bec3-8cc8eff5196e" class="cell" data-execution_count="1">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb1"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> pandas <span class="im">as</span> pd</span>
<span id="cb1-2"><a href="#cb1-2" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> numpy <span class="im">as</span> np</span>
<span id="cb1-3"><a href="#cb1-3" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> os</span>
<span id="cb1-4"><a href="#cb1-4" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.ensemble <span class="im">import</span> RandomForestClassifier, GradientBoostingClassifier</span>
<span id="cb1-5"><a href="#cb1-5" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.metrics <span class="im">import</span> roc_auc_score, accuracy_score</span>
<span id="cb1-6"><a href="#cb1-6" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.utils.class_weight <span class="im">import</span> compute_class_weight</span>
<span id="cb1-7"><a href="#cb1-7" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.model_selection <span class="im">import</span> GridSearchCV, train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV</span>
<span id="cb1-8"><a href="#cb1-8" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.datasets <span class="im">import</span> make_classification</span>
<span id="cb1-9"><a href="#cb1-9" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.inspection <span class="im">import</span> PartialDependenceDisplay</span>
<span id="cb1-10"><a href="#cb1-10" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> xgboost <span class="im">import</span> XGBClassifier</span>
<span id="cb1-11"><a href="#cb1-11" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb1-12"><a href="#cb1-12" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb1-13"><a href="#cb1-13" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> shap</span>
<span id="cb1-14"><a href="#cb1-14" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> catboost <span class="im">import</span> CatBoostClassifier</span>
<span id="cb1-15"><a href="#cb1-15" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> matplotlib.pyplot <span class="im">as</span> plt</span>
<span id="cb1-16"><a href="#cb1-16" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> joblib <span class="im">import</span> dump, load</span>
<span id="cb1-17"><a href="#cb1-17" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb1-18"><a href="#cb1-18" aria-hidden="true" tabindex="-1"></a>pd.set_option(<span class="st">"display.max_columns"</span>, <span class="va">None</span>)</span>
<span id="cb1-19"><a href="#cb1-19" aria-hidden="true" tabindex="-1"></a>pd.set_option(<span class="st">"display.max_rows"</span>, <span class="va">None</span>)</span>
<span id="cb1-20"><a href="#cb1-20" aria-hidden="true" tabindex="-1"></a>pd.options.display.float_format <span class="op">=</span> <span class="st">"</span><span class="sc">{:,.4}</span><span class="st">"</span>.<span class="bu">format</span></span>
<span id="cb1-21"><a href="#cb1-21" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb1-22"><a href="#cb1-22" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb1-23"><a href="#cb1-23" aria-hidden="true" tabindex="-1"></a><span class="co"># Run H2O later</span></span>
<span id="cb1-24"><a href="#cb1-24" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> h2o</span>
<span id="cb1-25"><a href="#cb1-25" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> pandoc</span>
<span id="cb1-26"><a href="#cb1-26" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> h2o.estimators.glm <span class="im">import</span> H2OGeneralizedLinearEstimator</span>
<span id="cb1-27"><a href="#cb1-27" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> h2o.estimators.gbm <span class="im">import</span> H2OGradientBoostingEstimator</span>
<span id="cb1-28"><a href="#cb1-28" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> h2o.estimators <span class="im">import</span> H2ORandomForestEstimator, H2ODecisionTreeEstimator</span>
<span id="cb1-29"><a href="#cb1-29" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> h2o.grid.grid_search <span class="im">import</span> H2OGridSearch</span>
<span id="cb1-30"><a href="#cb1-30" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> h2o.estimators <span class="im">import</span> H2OKMeansEstimator</span>
<span id="cb1-31"><a href="#cb1-31" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> h2o.automl <span class="im">import</span> H2OAutoML</span>
<span id="cb1-32"><a href="#cb1-32" aria-hidden="true" tabindex="-1"></a>h2o.init(max_mem_size <span class="op">=</span> <span class="st">"20G"</span>) <span class="co"># need more memory</span></span>
<span id="cb1-33"><a href="#cb1-33" aria-hidden="true" tabindex="-1"></a><span class="co">#h2o.init() # need more memory</span></span>
<span id="cb1-34"><a href="#cb1-34" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb1-35"><a href="#cb1-35" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> warnings</span>
<span id="cb1-36"><a href="#cb1-36" aria-hidden="true" tabindex="-1"></a><span class="co"># Suppress all warnings</span></span>
<span id="cb1-37"><a href="#cb1-37" aria-hidden="true" tabindex="-1"></a>warnings.filterwarnings(<span class="st">"ignore"</span>)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-stdout">
<pre><code>Checking whether there is an H2O instance running at http://localhost:54321..... not found.
Attempting to start a local H2O server...
  Java Version: openjdk version "11.0.23" 2024-04-16 LTS; OpenJDK Runtime Environment Zulu11.72+19-CA (build 11.0.23+9-LTS); OpenJDK 64-Bit Server VM Zulu11.72+19-CA (build 11.0.23+9-LTS, mixed mode)
  Starting server from /opt/anaconda3/lib/python3.12/site-packages/h2o/backend/bin/h2o.jar
  Ice root: /var/folders/z5/xxddnyr978bb97d94x1wj3780000gn/T/tmp96srg0t1
  JVM stdout: /var/folders/z5/xxddnyr978bb97d94x1wj3780000gn/T/tmp96srg0t1/h2o_luis_started_from_python.out
  JVM stderr: /var/folders/z5/xxddnyr978bb97d94x1wj3780000gn/T/tmp96srg0t1/h2o_luis_started_from_python.err
  Server is running at http://127.0.0.1:54321
Connecting to H2O server at http://127.0.0.1:54321 ... successful.</code></pre>
</div>
<div class="cell-output cell-output-display">

<style>

#h2o-table-1.h2o-container {
  overflow-x: auto;
}
#h2o-table-1 .h2o-table {
  /* width: 100%; */
  margin-top: 1em;
  margin-bottom: 1em;
}
#h2o-table-1 .h2o-table caption {
  white-space: nowrap;
  caption-side: top;
  text-align: left;
  /* margin-left: 1em; */
  margin: 0;
  font-size: larger;
}
#h2o-table-1 .h2o-table thead {
  white-space: nowrap; 
  position: sticky;
  top: 0;
  box-shadow: 0 -1px inset;
}
#h2o-table-1 .h2o-table tbody {
  overflow: auto;
}
#h2o-table-1 .h2o-table th,
#h2o-table-1 .h2o-table td {
  text-align: right;
  /* border: 1px solid; */
}
#h2o-table-1 .h2o-table tr:nth-child(even) {
  /* background: #F5F5F5 */
}

</style>      
<div id="h2o-table-1" class="h2o-container">
  
<table class="h2o-table caption-top table table-sm table-striped small" data-quarto-postprocess="true">
<tbody>
<tr class="odd">
<td>H2O_cluster_uptime:</td>
<td>02 secs</td>
</tr>
<tr class="even">
<td>H2O_cluster_timezone:</td>
<td>America/New_York</td>
</tr>
<tr class="odd">
<td>H2O_data_parsing_timezone:</td>
<td>UTC</td>
</tr>
<tr class="even">
<td>H2O_cluster_version:</td>
<td>3.46.0.6</td>
</tr>
<tr class="odd">
<td>H2O_cluster_version_age:</td>
<td>16 days</td>
</tr>
<tr class="even">
<td>H2O_cluster_name:</td>
<td>H2O_from_python_luis_oovuc7</td>
</tr>
<tr class="odd">
<td>H2O_cluster_total_nodes:</td>
<td>1</td>
</tr>
<tr class="even">
<td>H2O_cluster_free_memory:</td>
<td>20 Gb</td>
</tr>
<tr class="odd">
<td>H2O_cluster_total_cores:</td>
<td>10</td>
</tr>
<tr class="even">
<td>H2O_cluster_allowed_cores:</td>
<td>10</td>
</tr>
<tr class="odd">
<td>H2O_cluster_status:</td>
<td>locked, healthy</td>
</tr>
<tr class="even">
<td>H2O_connection_url:</td>
<td>http://127.0.0.1:54321</td>
</tr>
<tr class="odd">
<td>H2O_connection_proxy:</td>
<td>{"http": null, "https": null}</td>
</tr>
<tr class="even">
<td>H2O_internal_security:</td>
<td>False</td>
</tr>
<tr class="odd">
<td>Python_version:</td>
<td>3.12.2 final</td>
</tr>
</tbody>
</table>

</div>
</div>
</div>
<div id="a8707e21-1704-44ac-8a29-16edf4bd8db0" class="cell" data-execution_count="3">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb3"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb3-1"><a href="#cb3-1" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> firingRate(df):</span>
<span id="cb3-2"><a href="#cb3-2" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> <span class="dv">1</span><span class="op">-</span>df.isna().<span class="bu">sum</span>(axis<span class="op">=</span><span class="dv">0</span>)<span class="op">/</span>df.shape[<span class="dv">0</span>]</span>
<span id="cb3-3"><a href="#cb3-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb3-4"><a href="#cb3-4" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> uniqueCount(df):</span>
<span id="cb3-5"><a href="#cb3-5" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> df.nunique()</span>
<span id="cb3-6"><a href="#cb3-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb3-7"><a href="#cb3-7" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> uniqueRate(df):</span>
<span id="cb3-8"><a href="#cb3-8" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> df.nunique()<span class="op">/</span>df.shape[<span class="dv">0</span>]</span>
<span id="cb3-9"><a href="#cb3-9" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb3-10"><a href="#cb3-10" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> firingRateDetailed(df):</span>
<span id="cb3-11"><a href="#cb3-11" aria-hidden="true" tabindex="-1"></a>    output <span class="op">=</span> []</span>
<span id="cb3-12"><a href="#cb3-12" aria-hidden="true" tabindex="-1"></a>    firingRate <span class="op">=</span> <span class="dv">1</span><span class="op">-</span>df.isna().<span class="bu">sum</span>(axis<span class="op">=</span><span class="dv">0</span>)<span class="op">/</span>df.shape[<span class="dv">0</span>]</span>
<span id="cb3-13"><a href="#cb3-13" aria-hidden="true" tabindex="-1"></a>    uniqueCount <span class="op">=</span> df.nunique()</span>
<span id="cb3-14"><a href="#cb3-14" aria-hidden="true" tabindex="-1"></a>    uniqueRate <span class="op">=</span> df.nunique()<span class="op">/</span>df.shape[<span class="dv">0</span>]</span>
<span id="cb3-15"><a href="#cb3-15" aria-hidden="true" tabindex="-1"></a>    <span class="cf">for</span> i <span class="kw">in</span> df.columns:</span>
<span id="cb3-16"><a href="#cb3-16" aria-hidden="true" tabindex="-1"></a>        <span class="cf">if</span> df[i].dtypes <span class="op">==</span> <span class="st">'O'</span>:</span>
<span id="cb3-17"><a href="#cb3-17" aria-hidden="true" tabindex="-1"></a>            <span class="cf">if</span> df[i].nunique() <span class="op">&lt;=</span> <span class="dv">15</span>:</span>
<span id="cb3-18"><a href="#cb3-18" aria-hidden="true" tabindex="-1"></a>                output.append(df[i].value_counts(dropna <span class="op">=</span> <span class="va">False</span>, normalize <span class="op">=</span> <span class="va">True</span>))</span>
<span id="cb3-19"><a href="#cb3-19" aria-hidden="true" tabindex="-1"></a>            <span class="cf">else</span>:</span>
<span id="cb3-20"><a href="#cb3-20" aria-hidden="true" tabindex="-1"></a>                output.append((i<span class="op">+</span><span class="st">': Categorical Column with '</span><span class="op">+</span> <span class="bu">str</span>(df[i].nunique()) <span class="op">+</span> <span class="st">' groups'</span>))</span>
<span id="cb3-21"><a href="#cb3-21" aria-hidden="true" tabindex="-1"></a>        <span class="cf">elif</span> (df[i].dtypes <span class="kw">in</span> [<span class="st">'int8'</span>,<span class="st">'int16'</span>,<span class="st">'int32'</span>,<span class="st">'int64'</span>,<span class="st">'float32'</span>,<span class="st">'float64'</span>]):</span>
<span id="cb3-22"><a href="#cb3-22" aria-hidden="true" tabindex="-1"></a>            output.append(df[i].describe())</span>
<span id="cb3-23"><a href="#cb3-23" aria-hidden="true" tabindex="-1"></a>        <span class="cf">else</span>:</span>
<span id="cb3-24"><a href="#cb3-24" aria-hidden="true" tabindex="-1"></a>            df[i] <span class="op">=</span> df[i].astype(<span class="bu">str</span>)</span>
<span id="cb3-25"><a href="#cb3-25" aria-hidden="true" tabindex="-1"></a>            <span class="cf">if</span> df[i].nunique() <span class="op">&lt;=</span> <span class="dv">15</span>:</span>
<span id="cb3-26"><a href="#cb3-26" aria-hidden="true" tabindex="-1"></a>                output.append(df[i].value_counts(dropna <span class="op">=</span> <span class="va">False</span>, normalize <span class="op">=</span> <span class="va">True</span>))</span>
<span id="cb3-27"><a href="#cb3-27" aria-hidden="true" tabindex="-1"></a>            <span class="cf">else</span>:</span>
<span id="cb3-28"><a href="#cb3-28" aria-hidden="true" tabindex="-1"></a>                output.append((i<span class="op">+</span><span class="st">': Categorical Column with '</span><span class="op">+</span> <span class="bu">str</span>(df[i].nunique()) <span class="op">+</span> <span class="st">' groups'</span>))</span>
<span id="cb3-29"><a href="#cb3-29" aria-hidden="true" tabindex="-1"></a>    final <span class="op">=</span> pd.DataFrame({<span class="st">'Column'</span>: df.columns,</span>
<span id="cb3-30"><a href="#cb3-30" aria-hidden="true" tabindex="-1"></a>                          <span class="st">'firingRate'</span>: firingRate,</span>
<span id="cb3-31"><a href="#cb3-31" aria-hidden="true" tabindex="-1"></a>                          <span class="st">'uniqueCount'</span>: uniqueCount,</span>
<span id="cb3-32"><a href="#cb3-32" aria-hidden="true" tabindex="-1"></a>                          <span class="st">'uniqueRate'</span>: uniqueRate,</span>
<span id="cb3-33"><a href="#cb3-33" aria-hidden="true" tabindex="-1"></a>                          <span class="st">'Detailed_Summary'</span>: output})   </span>
<span id="cb3-34"><a href="#cb3-34" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span>(final)</span>
<span id="cb3-35"><a href="#cb3-35" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb3-36"><a href="#cb3-36" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> compute_class_weights(y):</span>
<span id="cb3-37"><a href="#cb3-37" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Calculate the class weights</span></span>
<span id="cb3-38"><a href="#cb3-38" aria-hidden="true" tabindex="-1"></a>    class_weights <span class="op">=</span> compute_class_weight(</span>
<span id="cb3-39"><a href="#cb3-39" aria-hidden="true" tabindex="-1"></a>        class_weight<span class="op">=</span><span class="st">"balanced"</span>, classes<span class="op">=</span>np.unique(y), y<span class="op">=</span>y</span>
<span id="cb3-40"><a href="#cb3-40" aria-hidden="true" tabindex="-1"></a>    )</span>
<span id="cb3-41"><a href="#cb3-41" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb3-42"><a href="#cb3-42" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Convert the result to a dictionary with class labels as keys</span></span>
<span id="cb3-43"><a href="#cb3-43" aria-hidden="true" tabindex="-1"></a>    class_weight_dict <span class="op">=</span> <span class="bu">dict</span>(<span class="bu">zip</span>(np.unique(y), class_weights))</span>
<span id="cb3-44"><a href="#cb3-44" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb3-45"><a href="#cb3-45" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Create an array of the same length as y with the corresponding class weight for each element</span></span>
<span id="cb3-46"><a href="#cb3-46" aria-hidden="true" tabindex="-1"></a>    class_weights_array <span class="op">=</span> np.array([class_weight_dict[label] <span class="cf">for</span> label <span class="kw">in</span> y])</span>
<span id="cb3-47"><a href="#cb3-47" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb3-48"><a href="#cb3-48" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> class_weights_array</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
</div>
<section id="read-in-files" class="level2">
<h2 class="anchored" data-anchor-id="read-in-files">Read in files</h2>
<div id="db504fec-8e89-4519-bdfd-abed86c02be8" class="cell" data-execution_count="5">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb4"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb4-1"><a href="#cb4-1" aria-hidden="true" tabindex="-1"></a>df <span class="op">=</span> pd.read_parquet(<span class="st">"giggle_user_ids_33603_w2flags_20241116.parquet"</span>)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
</div>
</section>
<section id="create-samples" class="level2">
<h2 class="anchored" data-anchor-id="create-samples">Create Samples</h2>
<section id="class-0---non-w2" class="level3">
<h3 class="anchored" data-anchor-id="class-0---non-w2">Class 0 - Non W2</h3>
</section>
<section id="class-1---w2" class="level3">
<h3 class="anchored" data-anchor-id="class-1---w2">Class 1 - W2</h3>
<div id="e1c6e4d4-627d-4314-94a2-e7af66d72069" class="cell" data-execution_count="7">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb5"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb5-1"><a href="#cb5-1" aria-hidden="true" tabindex="-1"></a>cl0 <span class="op">=</span> df[df[<span class="st">'W2_INCOME'</span>]<span class="op">==</span><span class="dv">0</span>]</span>
<span id="cb5-2"><a href="#cb5-2" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(cl0.shape)</span>
<span id="cb5-3"><a href="#cb5-3" aria-hidden="true" tabindex="-1"></a>cl1 <span class="op">=</span> df[df[<span class="st">'W2_INCOME'</span>]<span class="op">==</span><span class="dv">1</span>]</span>
<span id="cb5-4"><a href="#cb5-4" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(cl1.shape)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-stdout">
<pre><code>(33261, 5021)
(342, 5021)</code></pre>
</div>
</div>
<div id="a0c31077-4df1-4e25-a554-8ea209c7defc" class="cell" data-execution_count="9">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb7"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb7-1"><a href="#cb7-1" aria-hidden="true" tabindex="-1"></a>train0 <span class="op">=</span> cl0.sample(<span class="bu">int</span>(<span class="bu">len</span>(cl0)<span class="op">*</span><span class="fl">.7</span>), random_state <span class="op">=</span> <span class="dv">181</span>, replace <span class="op">=</span> <span class="va">False</span>)</span>
<span id="cb7-2"><a href="#cb7-2" aria-hidden="true" tabindex="-1"></a>test0 <span class="op">=</span> cl0[<span class="op">~</span>cl0.USER_ID.isin(train0.USER_ID.tolist())]</span>
<span id="cb7-3"><a href="#cb7-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb7-4"><a href="#cb7-4" aria-hidden="true" tabindex="-1"></a>train1 <span class="op">=</span> cl1.sample(<span class="bu">int</span>(<span class="bu">len</span>(cl1)<span class="op">*</span><span class="fl">.7</span>), random_state <span class="op">=</span> <span class="dv">181</span>, replace <span class="op">=</span> <span class="va">False</span>)</span>
<span id="cb7-5"><a href="#cb7-5" aria-hidden="true" tabindex="-1"></a>test1 <span class="op">=</span> cl1[<span class="op">~</span>cl1.USER_ID.isin(train1.USER_ID.tolist())]</span>
<span id="cb7-6"><a href="#cb7-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb7-7"><a href="#cb7-7" aria-hidden="true" tabindex="-1"></a>train <span class="op">=</span> pd.concat([train0, train1])</span>
<span id="cb7-8"><a href="#cb7-8" aria-hidden="true" tabindex="-1"></a>test <span class="op">=</span> pd.concat([test0, test1])</span>
<span id="cb7-9"><a href="#cb7-9" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb7-10"><a href="#cb7-10" aria-hidden="true" tabindex="-1"></a>train <span class="op">=</span> train.sample(<span class="bu">int</span>(<span class="bu">len</span>(train)), random_state <span class="op">=</span> <span class="dv">191</span>, replace <span class="op">=</span> <span class="va">False</span>)</span>
<span id="cb7-11"><a href="#cb7-11" aria-hidden="true" tabindex="-1"></a>test <span class="op">=</span> test.sample(<span class="bu">int</span>(<span class="bu">len</span>(test)), random_state <span class="op">=</span> <span class="dv">191</span>, replace <span class="op">=</span> <span class="va">False</span>)</span>
<span id="cb7-12"><a href="#cb7-12" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb7-13"><a href="#cb7-13" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">"Train Size: "</span> <span class="op">+</span> <span class="bu">str</span>(<span class="bu">len</span>(train)))</span>
<span id="cb7-14"><a href="#cb7-14" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">"Train W2's: "</span> <span class="op">+</span> <span class="bu">str</span>(<span class="bu">len</span>(train[train[<span class="st">'W2_INCOME'</span>] <span class="op">==</span> <span class="dv">1</span>])))</span>
<span id="cb7-15"><a href="#cb7-15" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">"Test Size: "</span> <span class="op">+</span> <span class="bu">str</span>(<span class="bu">len</span>(test)))</span>
<span id="cb7-16"><a href="#cb7-16" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">"Test W2's: "</span> <span class="op">+</span> <span class="bu">str</span>(<span class="bu">len</span>(test[test[<span class="st">'W2_INCOME'</span>] <span class="op">==</span> <span class="dv">1</span>])))</span>
<span id="cb7-17"><a href="#cb7-17" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb7-18"><a href="#cb7-18" aria-hidden="true" tabindex="-1"></a>train[<span class="st">'dataset'</span>] <span class="op">=</span> <span class="st">'train'</span></span>
<span id="cb7-19"><a href="#cb7-19" aria-hidden="true" tabindex="-1"></a>test[<span class="st">'dataset'</span>] <span class="op">=</span> <span class="st">'test'</span></span>
<span id="cb7-20"><a href="#cb7-20" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb7-21"><a href="#cb7-21" aria-hidden="true" tabindex="-1"></a>combine <span class="op">=</span> pd.concat([train, test])</span>
<span id="cb7-22"><a href="#cb7-22" aria-hidden="true" tabindex="-1"></a>combine <span class="op">=</span> combine.sample(<span class="bu">int</span>(<span class="bu">len</span>(combine)), random_state <span class="op">=</span> <span class="dv">191</span>, replace <span class="op">=</span> <span class="va">False</span>).reset_index(drop <span class="op">=</span> <span class="va">True</span>)</span>
<span id="cb7-23"><a href="#cb7-23" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">"Combined Data Size: "</span> <span class="op">+</span> <span class="bu">str</span>(<span class="bu">len</span>(combine)))</span>
<span id="cb7-24"><a href="#cb7-24" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb7-25"><a href="#cb7-25" aria-hidden="true" tabindex="-1"></a></span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-stdout">
<pre><code>Train Size: 23521
Train W2's: 239
Test Size: 10082
Test W2's: 103
Combined Data Size: 33603</code></pre>
</div>
</div>
</section>
</section>
<section id="list-of-predictorsnon-predictors" class="level2">
<h2 class="anchored" data-anchor-id="list-of-predictorsnon-predictors">List of predictors/non-predictors</h2>
<section id="read-in-feature-selection-results-display-top-10" class="level3">
<h3 class="anchored" data-anchor-id="read-in-feature-selection-results-display-top-10">Read in Feature Selection Results (display top 10)</h3>
<div id="54651126-cf40-43ca-a075-e6d5973f821f" class="cell" data-execution_count="11">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb9"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb9-1"><a href="#cb9-1" aria-hidden="true" tabindex="-1"></a>top_features <span class="op">=</span> pd.read_csv(<span class="st">"Feature_Selection_84_20241118.csv"</span>)</span>
<span id="cb9-2"><a href="#cb9-2" aria-hidden="true" tabindex="-1"></a>top_features.head(<span class="dv">10</span>)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-display" data-execution_count="11">
<div>


<table class="dataframe caption-top table table-sm table-striped small" data-quarto-postprocess="true" data-border="1">
<thead>
<tr class="header">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Variable</th>
<th data-quarto-table-cell-role="th">Total Relative Importance</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td data-quarto-table-cell-role="th">0</td>
<td>EARNED_WAGE_ACCESS_COUNT_TREND_MONTH_3</td>
<td>6.003e+04</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">1</td>
<td>EARNED_WAGE_ACCESS_COUNT_PAST_180D</td>
<td>5.892e+04</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">2</td>
<td>EARNED_WAGE_ACCESS_COUNT_PAST_270D</td>
<td>4.417e+04</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">3</td>
<td>EARNED_WAGE_ACCESS_COUNT_TREND_MONTH_1</td>
<td>4.019e+04</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">4</td>
<td>GIG_INCOME_TO_INCOME_RATIO_PAST_180D</td>
<td>3.797e+04</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">5</td>
<td>EARNED_WAGE_ACCESS_TREND_MONTH_1</td>
<td>3.637e+04</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">6</td>
<td>INCOME_NEXT_7D</td>
<td>3.605e+04</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">7</td>
<td>EARNED_WAGE_ACCESS_TREND_MONTH_3</td>
<td>3.582e+04</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">8</td>
<td>INCOME_NEXT_30D</td>
<td>3.446e+04</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">9</td>
<td>RENT_TO_INCOME_RATIO_PAST_360D</td>
<td>3.119e+04</td>
</tr>
</tbody>
</table>

</div>
</div>
</div>
<div id="bbc6f1b3-2351-4e0a-8d88-c7e58993256c" class="cell" data-execution_count="13">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb10"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb10-1"><a href="#cb10-1" aria-hidden="true" tabindex="-1"></a>predictors <span class="op">=</span> top_features.Variable.tolist()</span>
<span id="cb10-2"><a href="#cb10-2" aria-hidden="true" tabindex="-1"></a>outcome_flag <span class="op">=</span> <span class="st">'W2_INCOME'</span></span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
</div>
</section>
</section>
<section id="these-will-be-columns-to-keep-in-h2o-frame-show-last-5" class="level2">
<h2 class="anchored" data-anchor-id="these-will-be-columns-to-keep-in-h2o-frame-show-last-5">These will be columns to keep in H2O Frame (show last 5)</h2>
<div id="8438aa99-4492-41f2-9d8c-979a4981b059" class="cell" data-execution_count="15">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb11"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb11-1"><a href="#cb11-1" aria-hidden="true" tabindex="-1"></a>col_keep <span class="op">=</span> predictors<span class="op">+</span>[<span class="st">'W2_INCOME'</span>, <span class="st">'dataset'</span>]</span>
<span id="cb11-2"><a href="#cb11-2" aria-hidden="true" tabindex="-1"></a>col_keep[<span class="dv">80</span>:<span class="dv">85</span>]</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-display" data-execution_count="15">
<pre><code>['MEDIAN_BNPL_PAYMENT_PAST_30D',
 'DEBT_PMT_STD_PAST_2Y',
 'MEDIAN_BNPL_LOAN_AMOUNT_PAST_90D',
 'OBLIGATORY_OUTFLOWS_TO_OUTFLOWS_RATIO_PAST_90D',
 'W2_INCOME']</code></pre>
</div>
</div>
</section>
<section id="convert-dataframe-to-h2o-frame" class="level2">
<h2 class="anchored" data-anchor-id="convert-dataframe-to-h2o-frame">Convert DataFrame to H2O Frame</h2>
<div id="ded2f1f1-72f7-48b9-a4a1-c310427db333" class="cell" data-execution_count="17">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb13"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb13-1"><a href="#cb13-1" aria-hidden="true" tabindex="-1"></a>df_h2o <span class="op">=</span> h2o.H2OFrame(combine[col_keep])</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-stdout">
<pre><code>Parse progress: |████████████████████████████████████████████████████████████████| (done) 100%</code></pre>
</div>
</div>
<div id="b64a321f-99e4-46d3-944e-0400e027fba2" class="cell" data-execution_count="19">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb15"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb15-1"><a href="#cb15-1" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> time</span>
<span id="cb15-2"><a href="#cb15-2" aria-hidden="true" tabindex="-1"></a>start <span class="op">=</span> time.time()</span>
<span id="cb15-3"><a href="#cb15-3" aria-hidden="true" tabindex="-1"></a><span class="co"># Fix Column Types in h2o</span></span>
<span id="cb15-4"><a href="#cb15-4" aria-hidden="true" tabindex="-1"></a><span class="co"># If column enum has "9 or more" values, turn numeric (to prevent high cardinality in categorical)</span></span>
<span id="cb15-5"><a href="#cb15-5" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> i <span class="kw">in</span> df_h2o.columns:</span>
<span id="cb15-6"><a href="#cb15-6" aria-hidden="true" tabindex="-1"></a>    <span class="cf">if</span> (df_h2o.types[i] <span class="op">==</span> <span class="st">"enum"</span>):</span>
<span id="cb15-7"><a href="#cb15-7" aria-hidden="true" tabindex="-1"></a>        <span class="cf">if</span> <span class="bu">len</span>(df_h2o[i].levels()[<span class="dv">0</span>]) <span class="op">&gt;=</span> <span class="dv">9</span>:</span>
<span id="cb15-8"><a href="#cb15-8" aria-hidden="true" tabindex="-1"></a>            df_h2o[i] <span class="op">=</span> df_h2o[i].asnumeric()</span>
<span id="cb15-9"><a href="#cb15-9" aria-hidden="true" tabindex="-1"></a>            </span>
<span id="cb15-10"><a href="#cb15-10" aria-hidden="true" tabindex="-1"></a>    <span class="co">#elif df_h2o.types[i] == "int":</span></span>
<span id="cb15-11"><a href="#cb15-11" aria-hidden="true" tabindex="-1"></a>    <span class="co">#    df_h2o[i] = df_h2o[i].asnumeric()</span></span>
<span id="cb15-12"><a href="#cb15-12" aria-hidden="true" tabindex="-1"></a>end <span class="op">=</span> time.time()</span>
<span id="cb15-13"><a href="#cb15-13" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">"Execution time:"</span>, end <span class="op">-</span> start, <span class="st">"seconds"</span>)</span>
<span id="cb15-14"><a href="#cb15-14" aria-hidden="true" tabindex="-1"></a><span class="co">#df_h2o.types</span></span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-stdout">
<pre><code>Execution time: 3.9899189472198486 seconds</code></pre>
</div>
</div>
<div id="45e02513-63eb-46d4-8662-3ab581971ec9" class="cell" data-execution_count="20">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb17"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb17-1"><a href="#cb17-1" aria-hidden="true" tabindex="-1"></a>df_h2o[<span class="st">"W2_INCOME"</span>] <span class="op">=</span> df_h2o[<span class="st">"W2_INCOME"</span>].asfactor()</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
</div>
<div id="3112e88b-5f93-4839-bb7a-5722cd1f94f1" class="cell" data-execution_count="21">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb18"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb18-1"><a href="#cb18-1" aria-hidden="true" tabindex="-1"></a>df_train <span class="op">=</span> df_h2o[df_h2o[<span class="st">'dataset'</span>] <span class="op">==</span> <span class="st">'train'</span>]</span>
<span id="cb18-2"><a href="#cb18-2" aria-hidden="true" tabindex="-1"></a>df_test <span class="op">=</span> df_h2o[df_h2o[<span class="st">'dataset'</span>] <span class="op">==</span> <span class="st">'test'</span>]</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
</div>
</section>
<section id="grid-search-of-models-using-random-forest-estimator" class="level2">
<h2 class="anchored" data-anchor-id="grid-search-of-models-using-random-forest-estimator">Grid Search of Models (Using Random Forest Estimator)</h2>
<div id="d568ea22-a18f-408c-9d7c-07ff063742ff" class="cell" data-execution_count="25">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb19"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb19-1"><a href="#cb19-1" aria-hidden="true" tabindex="-1"></a>drf_params1 <span class="op">=</span> {</span>
<span id="cb19-2"><a href="#cb19-2" aria-hidden="true" tabindex="-1"></a>    <span class="st">"max_depth"</span>: [<span class="dv">5</span>, <span class="dv">7</span>, <span class="dv">10</span>, <span class="dv">15</span>, <span class="dv">20</span>, <span class="dv">25</span>],               <span class="co"># Extended range</span></span>
<span id="cb19-3"><a href="#cb19-3" aria-hidden="true" tabindex="-1"></a>    <span class="st">"sample_rate"</span>: [<span class="fl">0.5</span>, <span class="fl">0.6</span>, <span class="fl">0.7</span>, <span class="fl">0.8</span>, <span class="fl">0.9</span>],         <span class="co"># More options for sampling</span></span>
<span id="cb19-4"><a href="#cb19-4" aria-hidden="true" tabindex="-1"></a>    <span class="st">"ntrees"</span>: [<span class="dv">10</span>, <span class="dv">15</span>, <span class="dv">20</span>, <span class="dv">25</span>, <span class="dv">30</span>, <span class="dv">35</span>, <span class="dv">40</span>, <span class="dv">50</span>],       <span class="co"># Larger range for tree count</span></span>
<span id="cb19-5"><a href="#cb19-5" aria-hidden="true" tabindex="-1"></a>    <span class="st">"min_rows"</span>: [<span class="dv">1</span>, <span class="dv">5</span>, <span class="dv">10</span>, <span class="dv">20</span>, <span class="dv">50</span>],                   <span class="co"># Smaller minimum rows for high granularity</span></span>
<span id="cb19-6"><a href="#cb19-6" aria-hidden="true" tabindex="-1"></a>    <span class="st">"mtries"</span>: [<span class="dv">1</span>, <span class="dv">3</span>, <span class="dv">5</span>, <span class="op">-</span><span class="dv">1</span>],                          <span class="co"># Add '-1' for auto (sqrt of predictors)</span></span>
<span id="cb19-7"><a href="#cb19-7" aria-hidden="true" tabindex="-1"></a>    <span class="st">"nbins"</span>: [<span class="dv">10</span>, <span class="dv">20</span>, <span class="dv">30</span>, <span class="dv">40</span>, <span class="dv">50</span>],                    <span class="co"># Number of bins for histogram splitting</span></span>
<span id="cb19-8"><a href="#cb19-8" aria-hidden="true" tabindex="-1"></a>    <span class="st">"nbins_top_level"</span>: [<span class="dv">50</span>, <span class="dv">100</span>, <span class="dv">200</span>],                <span class="co"># Top-level histogram bins</span></span>
<span id="cb19-9"><a href="#cb19-9" aria-hidden="true" tabindex="-1"></a>    <span class="st">"nbins_cats"</span>: [<span class="dv">16</span>, <span class="dv">64</span>, <span class="dv">256</span>],                      <span class="co"># For categorical predictors</span></span>
<span id="cb19-10"><a href="#cb19-10" aria-hidden="true" tabindex="-1"></a>    <span class="st">"col_sample_rate_per_tree"</span>: [<span class="fl">0.5</span>, <span class="fl">0.6</span>, <span class="fl">0.7</span>, <span class="fl">0.8</span>], <span class="co"># Column sampling for each tree</span></span>
<span id="cb19-11"><a href="#cb19-11" aria-hidden="true" tabindex="-1"></a>    <span class="st">"distribution"</span>: [<span class="st">"bernoulli"</span>],     <span class="co"># Add support for other distributions if applicable</span></span>
<span id="cb19-12"><a href="#cb19-12" aria-hidden="true" tabindex="-1"></a>    <span class="st">"seed"</span>: [<span class="dv">1234</span>]                              <span class="co"># Add seeds for reproducibility</span></span>
<span id="cb19-13"><a href="#cb19-13" aria-hidden="true" tabindex="-1"></a>}</span>
<span id="cb19-14"><a href="#cb19-14" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-15"><a href="#cb19-15" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-16"><a href="#cb19-16" aria-hidden="true" tabindex="-1"></a>search_criteria <span class="op">=</span> {</span>
<span id="cb19-17"><a href="#cb19-17" aria-hidden="true" tabindex="-1"></a>    <span class="st">'strategy'</span>: <span class="st">"RandomDiscrete"</span>,</span>
<span id="cb19-18"><a href="#cb19-18" aria-hidden="true" tabindex="-1"></a>    <span class="st">'max_models'</span>: <span class="dv">100</span>,             <span class="co"># Limit runtime to 20 minutes</span></span>
<span id="cb19-19"><a href="#cb19-19" aria-hidden="true" tabindex="-1"></a>    <span class="st">'max_runtime_secs'</span>: <span class="dv">1200</span>,</span>
<span id="cb19-20"><a href="#cb19-20" aria-hidden="true" tabindex="-1"></a>    <span class="st">'stopping_metric'</span>: <span class="st">"AUC"</span>,             <span class="co"># Stop based on AUC</span></span>
<span id="cb19-21"><a href="#cb19-21" aria-hidden="true" tabindex="-1"></a>    <span class="st">'stopping_rounds'</span>: <span class="dv">5</span>,</span>
<span id="cb19-22"><a href="#cb19-22" aria-hidden="true" tabindex="-1"></a>    <span class="st">'stopping_tolerance'</span>: <span class="fl">0.001</span>,</span>
<span id="cb19-23"><a href="#cb19-23" aria-hidden="true" tabindex="-1"></a>    <span class="st">'seed'</span>: <span class="dv">1234</span> </span>
<span id="cb19-24"><a href="#cb19-24" aria-hidden="true" tabindex="-1"></a>}</span>
<span id="cb19-25"><a href="#cb19-25" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-26"><a href="#cb19-26" aria-hidden="true" tabindex="-1"></a><span class="co"># Initialize the Random Forest estimator with balance_classes enabled</span></span>
<span id="cb19-27"><a href="#cb19-27" aria-hidden="true" tabindex="-1"></a>rf_model <span class="op">=</span> H2ORandomForestEstimator(</span>
<span id="cb19-28"><a href="#cb19-28" aria-hidden="true" tabindex="-1"></a>    seed<span class="op">=</span><span class="dv">1</span>,</span>
<span id="cb19-29"><a href="#cb19-29" aria-hidden="true" tabindex="-1"></a>    nfolds <span class="op">=</span> <span class="dv">5</span>,</span>
<span id="cb19-30"><a href="#cb19-30" aria-hidden="true" tabindex="-1"></a>    balance_classes<span class="op">=</span><span class="va">True</span>   <span class="co"># Enables class balancing for imbalanced data</span></span>
<span id="cb19-31"><a href="#cb19-31" aria-hidden="true" tabindex="-1"></a>)</span>
<span id="cb19-32"><a href="#cb19-32" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-33"><a href="#cb19-33" aria-hidden="true" tabindex="-1"></a><span class="co"># Create and run the grid search</span></span>
<span id="cb19-34"><a href="#cb19-34" aria-hidden="true" tabindex="-1"></a>rf_grid <span class="op">=</span> H2OGridSearch(</span>
<span id="cb19-35"><a href="#cb19-35" aria-hidden="true" tabindex="-1"></a>    model<span class="op">=</span>rf_model,</span>
<span id="cb19-36"><a href="#cb19-36" aria-hidden="true" tabindex="-1"></a>    hyper_params<span class="op">=</span>drf_params1,</span>
<span id="cb19-37"><a href="#cb19-37" aria-hidden="true" tabindex="-1"></a>    grid_id<span class="op">=</span><span class="st">'drf_grid1'</span>,</span>
<span id="cb19-38"><a href="#cb19-38" aria-hidden="true" tabindex="-1"></a>    search_criteria<span class="op">=</span>search_criteria</span>
<span id="cb19-39"><a href="#cb19-39" aria-hidden="true" tabindex="-1"></a>)</span>
<span id="cb19-40"><a href="#cb19-40" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-41"><a href="#cb19-41" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-42"><a href="#cb19-42" aria-hidden="true" tabindex="-1"></a>rf_grid.train(x<span class="op">=</span>predictors, </span>
<span id="cb19-43"><a href="#cb19-43" aria-hidden="true" tabindex="-1"></a>                y<span class="op">=</span>outcome_flag,</span>
<span id="cb19-44"><a href="#cb19-44" aria-hidden="true" tabindex="-1"></a>                training_frame<span class="op">=</span> df_train,</span>
<span id="cb19-45"><a href="#cb19-45" aria-hidden="true" tabindex="-1"></a>              validation_frame<span class="op">=</span> df_test,</span>
<span id="cb19-46"><a href="#cb19-46" aria-hidden="true" tabindex="-1"></a>                seed<span class="op">=</span><span class="dv">1</span>)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-stdout">
<pre><code>drf Grid Build progress: |███████████████████████████████████████████████████████| (done) 100%</code></pre>
</div>
<div class="cell-output cell-output-display" data-execution_count="25">

<style>

#h2o-table-2.h2o-container {
  overflow-x: auto;
}
#h2o-table-2 .h2o-table {
  /* width: 100%; */
  margin-top: 1em;
  margin-bottom: 1em;
}
#h2o-table-2 .h2o-table caption {
  white-space: nowrap;
  caption-side: top;
  text-align: left;
  /* margin-left: 1em; */
  margin: 0;
  font-size: larger;
}
#h2o-table-2 .h2o-table thead {
  white-space: nowrap; 
  position: sticky;
  top: 0;
  box-shadow: 0 -1px inset;
}
#h2o-table-2 .h2o-table tbody {
  overflow: auto;
}
#h2o-table-2 .h2o-table th,
#h2o-table-2 .h2o-table td {
  text-align: right;
  /* border: 1px solid; */
}
#h2o-table-2 .h2o-table tr:nth-child(even) {
  /* background: #F5F5F5 */
}

</style>      
<div id="h2o-table-2" class="h2o-container">
  
<table class="h2o-table caption-top table table-sm table-striped small" data-quarto-postprocess="true">
<caption>Hyper-Parameter Search Summary: ordered by increasing logloss</caption>
<thead>
<tr class="header">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">col_sample_rate_per_tree</th>
<th data-quarto-table-cell-role="th">distribution</th>
<th data-quarto-table-cell-role="th">max_depth</th>
<th data-quarto-table-cell-role="th">min_rows</th>
<th data-quarto-table-cell-role="th">mtries</th>
<th data-quarto-table-cell-role="th">nbins</th>
<th data-quarto-table-cell-role="th">nbins_cats</th>
<th data-quarto-table-cell-role="th">nbins_top_level</th>
<th data-quarto-table-cell-role="th">ntrees</th>
<th data-quarto-table-cell-role="th">sample_rate</th>
<th data-quarto-table-cell-role="th">seed</th>
<th data-quarto-table-cell-role="th">model_ids</th>
<th data-quarto-table-cell-role="th">logloss</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td></td>
<td>0.8</td>
<td>bernoulli</td>
<td>5.0</td>
<td>10.0</td>
<td>-1.0</td>
<td>10.0</td>
<td>256.0</td>
<td>200.0</td>
<td>35.0</td>
<td>0.6</td>
<td>1234.0</td>
<td>drf_grid1_model_43</td>
<td>0.0452794</td>
</tr>
<tr class="even">
<td></td>
<td>0.6</td>
<td>bernoulli</td>
<td>7.0</td>
<td>50.0</td>
<td>-1.0</td>
<td>10.0</td>
<td>16.0</td>
<td>100.0</td>
<td>15.0</td>
<td>0.5</td>
<td>1234.0</td>
<td>drf_grid1_model_86</td>
<td>0.0454753</td>
</tr>
<tr class="odd">
<td></td>
<td>0.6</td>
<td>bernoulli</td>
<td>5.0</td>
<td>1.0</td>
<td>-1.0</td>
<td>30.0</td>
<td>256.0</td>
<td>200.0</td>
<td>10.0</td>
<td>0.7</td>
<td>1234.0</td>
<td>drf_grid1_model_28</td>
<td>0.0455021</td>
</tr>
<tr class="even">
<td></td>
<td>0.6</td>
<td>bernoulli</td>
<td>5.0</td>
<td>20.0</td>
<td>-1.0</td>
<td>30.0</td>
<td>256.0</td>
<td>100.0</td>
<td>35.0</td>
<td>0.8</td>
<td>1234.0</td>
<td>drf_grid1_model_72</td>
<td>0.0455302</td>
</tr>
<tr class="odd">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>5.0</td>
<td>20.0</td>
<td>-1.0</td>
<td>20.0</td>
<td>64.0</td>
<td>50.0</td>
<td>40.0</td>
<td>0.8</td>
<td>1234.0</td>
<td>drf_grid1_model_49</td>
<td>0.0455320</td>
</tr>
<tr class="even">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>5.0</td>
<td>10.0</td>
<td>-1.0</td>
<td>50.0</td>
<td>64.0</td>
<td>50.0</td>
<td>50.0</td>
<td>0.6</td>
<td>1234.0</td>
<td>drf_grid1_model_94</td>
<td>0.0455838</td>
</tr>
<tr class="odd">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>5.0</td>
<td>10.0</td>
<td>-1.0</td>
<td>40.0</td>
<td>256.0</td>
<td>100.0</td>
<td>25.0</td>
<td>0.8</td>
<td>1234.0</td>
<td>drf_grid1_model_84</td>
<td>0.0459985</td>
</tr>
<tr class="even">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>5.0</td>
<td>50.0</td>
<td>5.0</td>
<td>40.0</td>
<td>64.0</td>
<td>50.0</td>
<td>50.0</td>
<td>0.6</td>
<td>1234.0</td>
<td>drf_grid1_model_26</td>
<td>0.0461046</td>
</tr>
<tr class="odd">
<td></td>
<td>0.6</td>
<td>bernoulli</td>
<td>5.0</td>
<td>5.0</td>
<td>5.0</td>
<td>10.0</td>
<td>16.0</td>
<td>200.0</td>
<td>30.0</td>
<td>0.5</td>
<td>1234.0</td>
<td>drf_grid1_model_35</td>
<td>0.0462889</td>
</tr>
<tr class="even">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>5.0</td>
<td>50.0</td>
<td>5.0</td>
<td>20.0</td>
<td>64.0</td>
<td>50.0</td>
<td>30.0</td>
<td>0.7</td>
<td>1234.0</td>
<td>drf_grid1_model_12</td>
<td>0.0463223</td>
</tr>
<tr class="odd">
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
</tr>
<tr class="even">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>25.0</td>
<td>10.0</td>
<td>5.0</td>
<td>40.0</td>
<td>64.0</td>
<td>50.0</td>
<td>15.0</td>
<td>0.8</td>
<td>1234.0</td>
<td>drf_grid1_model_31</td>
<td>0.0979471</td>
</tr>
<tr class="odd">
<td></td>
<td>0.8</td>
<td>bernoulli</td>
<td>25.0</td>
<td>5.0</td>
<td>-1.0</td>
<td>50.0</td>
<td>64.0</td>
<td>200.0</td>
<td>20.0</td>
<td>0.6</td>
<td>1234.0</td>
<td>drf_grid1_model_65</td>
<td>0.0980817</td>
</tr>
<tr class="even">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>20.0</td>
<td>1.0</td>
<td>5.0</td>
<td>20.0</td>
<td>256.0</td>
<td>200.0</td>
<td>30.0</td>
<td>0.6</td>
<td>1234.0</td>
<td>drf_grid1_model_88</td>
<td>0.1011293</td>
</tr>
<tr class="odd">
<td></td>
<td>0.6</td>
<td>bernoulli</td>
<td>20.0</td>
<td>5.0</td>
<td>5.0</td>
<td>30.0</td>
<td>256.0</td>
<td>50.0</td>
<td>20.0</td>
<td>0.9</td>
<td>1234.0</td>
<td>drf_grid1_model_77</td>
<td>0.1028425</td>
</tr>
<tr class="even">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>20.0</td>
<td>10.0</td>
<td>5.0</td>
<td>10.0</td>
<td>256.0</td>
<td>100.0</td>
<td>10.0</td>
<td>0.8</td>
<td>1234.0</td>
<td>drf_grid1_model_93</td>
<td>0.1041151</td>
</tr>
<tr class="odd">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>20.0</td>
<td>1.0</td>
<td>-1.0</td>
<td>30.0</td>
<td>64.0</td>
<td>200.0</td>
<td>25.0</td>
<td>0.6</td>
<td>1234.0</td>
<td>drf_grid1_model_51</td>
<td>0.1044277</td>
</tr>
<tr class="even">
<td></td>
<td>0.8</td>
<td>bernoulli</td>
<td>20.0</td>
<td>5.0</td>
<td>3.0</td>
<td>20.0</td>
<td>256.0</td>
<td>100.0</td>
<td>10.0</td>
<td>0.5</td>
<td>1234.0</td>
<td>drf_grid1_model_44</td>
<td>0.1104028</td>
</tr>
<tr class="odd">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>20.0</td>
<td>1.0</td>
<td>-1.0</td>
<td>40.0</td>
<td>16.0</td>
<td>50.0</td>
<td>15.0</td>
<td>0.5</td>
<td>1234.0</td>
<td>drf_grid1_model_57</td>
<td>0.1228886</td>
</tr>
<tr class="even">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>25.0</td>
<td>1.0</td>
<td>-1.0</td>
<td>40.0</td>
<td>64.0</td>
<td>100.0</td>
<td>20.0</td>
<td>0.7</td>
<td>1234.0</td>
<td>drf_grid1_model_1</td>
<td>0.1318061</td>
</tr>
<tr class="odd">
<td></td>
<td>0.6</td>
<td>bernoulli</td>
<td>25.0</td>
<td>1.0</td>
<td>-1.0</td>
<td>50.0</td>
<td>64.0</td>
<td>100.0</td>
<td>15.0</td>
<td>0.7</td>
<td>1234.0</td>
<td>drf_grid1_model_22</td>
<td>0.1440571</td>
</tr>
</tbody>
</table>

</div>
<pre style="font-size: smaller; margin-bottom: 1em;">[100 rows x 14 columns]</pre>
</div>
</div>
</section>
<section id="leaderboard-results" class="level2">
<h2 class="anchored" data-anchor-id="leaderboard-results">Leaderboard results</h2>
<div id="e7b130b5-c2ed-4027-93c5-fb654048962f" class="cell" data-scrolled="true" data-execution_count="28">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb21"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb21-1"><a href="#cb21-1" aria-hidden="true" tabindex="-1"></a>leaderboard <span class="op">=</span> rf_grid.get_grid(sort_by<span class="op">=</span><span class="st">'auc'</span>, decreasing<span class="op">=</span><span class="va">True</span>)</span>
<span id="cb21-2"><a href="#cb21-2" aria-hidden="true" tabindex="-1"></a>leaderboard</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-display" data-execution_count="28">

<style>

#h2o-table-3.h2o-container {
  overflow-x: auto;
}
#h2o-table-3 .h2o-table {
  /* width: 100%; */
  margin-top: 1em;
  margin-bottom: 1em;
}
#h2o-table-3 .h2o-table caption {
  white-space: nowrap;
  caption-side: top;
  text-align: left;
  /* margin-left: 1em; */
  margin: 0;
  font-size: larger;
}
#h2o-table-3 .h2o-table thead {
  white-space: nowrap; 
  position: sticky;
  top: 0;
  box-shadow: 0 -1px inset;
}
#h2o-table-3 .h2o-table tbody {
  overflow: auto;
}
#h2o-table-3 .h2o-table th,
#h2o-table-3 .h2o-table td {
  text-align: right;
  /* border: 1px solid; */
}
#h2o-table-3 .h2o-table tr:nth-child(even) {
  /* background: #F5F5F5 */
}

</style>      
<div id="h2o-table-3" class="h2o-container">
  
<table class="h2o-table caption-top table table-sm table-striped small" data-quarto-postprocess="true">
<caption>Hyper-Parameter Search Summary: ordered by decreasing auc</caption>
<thead>
<tr class="header">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">col_sample_rate_per_tree</th>
<th data-quarto-table-cell-role="th">distribution</th>
<th data-quarto-table-cell-role="th">max_depth</th>
<th data-quarto-table-cell-role="th">min_rows</th>
<th data-quarto-table-cell-role="th">mtries</th>
<th data-quarto-table-cell-role="th">nbins</th>
<th data-quarto-table-cell-role="th">nbins_cats</th>
<th data-quarto-table-cell-role="th">nbins_top_level</th>
<th data-quarto-table-cell-role="th">ntrees</th>
<th data-quarto-table-cell-role="th">sample_rate</th>
<th data-quarto-table-cell-role="th">seed</th>
<th data-quarto-table-cell-role="th">model_ids</th>
<th data-quarto-table-cell-role="th">auc</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>20.0</td>
<td>50.0</td>
<td>-1.0</td>
<td>10.0</td>
<td>256.0</td>
<td>100.0</td>
<td>30.0</td>
<td>0.8</td>
<td>1234.0</td>
<td>drf_grid1_model_78</td>
<td>0.9059074</td>
</tr>
<tr class="even">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>15.0</td>
<td>50.0</td>
<td>5.0</td>
<td>10.0</td>
<td>16.0</td>
<td>200.0</td>
<td>50.0</td>
<td>0.6</td>
<td>1234.0</td>
<td>drf_grid1_model_85</td>
<td>0.9052891</td>
</tr>
<tr class="odd">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>10.0</td>
<td>20.0</td>
<td>5.0</td>
<td>40.0</td>
<td>16.0</td>
<td>200.0</td>
<td>40.0</td>
<td>0.5</td>
<td>1234.0</td>
<td>drf_grid1_model_25</td>
<td>0.9047158</td>
</tr>
<tr class="even">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>15.0</td>
<td>50.0</td>
<td>-1.0</td>
<td>10.0</td>
<td>256.0</td>
<td>200.0</td>
<td>30.0</td>
<td>0.9</td>
<td>1234.0</td>
<td>drf_grid1_model_79</td>
<td>0.9042310</td>
</tr>
<tr class="odd">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>10.0</td>
<td>50.0</td>
<td>3.0</td>
<td>40.0</td>
<td>256.0</td>
<td>100.0</td>
<td>40.0</td>
<td>0.8</td>
<td>1234.0</td>
<td>drf_grid1_model_52</td>
<td>0.9017198</td>
</tr>
<tr class="even">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>10.0</td>
<td>50.0</td>
<td>-1.0</td>
<td>10.0</td>
<td>64.0</td>
<td>50.0</td>
<td>10.0</td>
<td>0.7</td>
<td>1234.0</td>
<td>drf_grid1_model_81</td>
<td>0.8988795</td>
</tr>
<tr class="odd">
<td></td>
<td>0.6</td>
<td>bernoulli</td>
<td>25.0</td>
<td>50.0</td>
<td>3.0</td>
<td>30.0</td>
<td>256.0</td>
<td>50.0</td>
<td>35.0</td>
<td>0.7</td>
<td>1234.0</td>
<td>drf_grid1_model_64</td>
<td>0.8935552</td>
</tr>
<tr class="even">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>7.0</td>
<td>20.0</td>
<td>-1.0</td>
<td>30.0</td>
<td>64.0</td>
<td>50.0</td>
<td>35.0</td>
<td>0.8</td>
<td>1234.0</td>
<td>drf_grid1_model_73</td>
<td>0.8934037</td>
</tr>
<tr class="odd">
<td></td>
<td>0.6</td>
<td>bernoulli</td>
<td>7.0</td>
<td>50.0</td>
<td>-1.0</td>
<td>10.0</td>
<td>16.0</td>
<td>100.0</td>
<td>15.0</td>
<td>0.5</td>
<td>1234.0</td>
<td>drf_grid1_model_86</td>
<td>0.8934011</td>
</tr>
<tr class="even">
<td></td>
<td>0.6</td>
<td>bernoulli</td>
<td>10.0</td>
<td>50.0</td>
<td>-1.0</td>
<td>20.0</td>
<td>256.0</td>
<td>100.0</td>
<td>10.0</td>
<td>0.8</td>
<td>1234.0</td>
<td>drf_grid1_model_21</td>
<td>0.8915046</td>
</tr>
<tr class="odd">
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
</tr>
<tr class="even">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>20.0</td>
<td>1.0</td>
<td>-1.0</td>
<td>30.0</td>
<td>64.0</td>
<td>200.0</td>
<td>25.0</td>
<td>0.6</td>
<td>1234.0</td>
<td>drf_grid1_model_51</td>
<td>0.8054194</td>
</tr>
<tr class="odd">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>20.0</td>
<td>10.0</td>
<td>5.0</td>
<td>10.0</td>
<td>256.0</td>
<td>100.0</td>
<td>10.0</td>
<td>0.8</td>
<td>1234.0</td>
<td>drf_grid1_model_93</td>
<td>0.7959275</td>
</tr>
<tr class="even">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>20.0</td>
<td>1.0</td>
<td>-1.0</td>
<td>40.0</td>
<td>16.0</td>
<td>50.0</td>
<td>15.0</td>
<td>0.5</td>
<td>1234.0</td>
<td>drf_grid1_model_57</td>
<td>0.7899624</td>
</tr>
<tr class="odd">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>20.0</td>
<td>5.0</td>
<td>3.0</td>
<td>20.0</td>
<td>64.0</td>
<td>200.0</td>
<td>15.0</td>
<td>0.9</td>
<td>1234.0</td>
<td>drf_grid1_model_13</td>
<td>0.7829220</td>
</tr>
<tr class="even">
<td></td>
<td>0.5</td>
<td>bernoulli</td>
<td>20.0</td>
<td>1.0</td>
<td>5.0</td>
<td>20.0</td>
<td>256.0</td>
<td>200.0</td>
<td>30.0</td>
<td>0.6</td>
<td>1234.0</td>
<td>drf_grid1_model_88</td>
<td>0.7784129</td>
</tr>
<tr class="odd">
<td></td>
<td>0.8</td>
<td>bernoulli</td>
<td>20.0</td>
<td>5.0</td>
<td>3.0</td>
<td>20.0</td>
<td>256.0</td>
<td>100.0</td>
<td>10.0</td>
<td>0.5</td>
<td>1234.0</td>
<td>drf_grid1_model_44</td>
<td>0.7667530</td>
</tr>
<tr class="even">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>25.0</td>
<td>1.0</td>
<td>-1.0</td>
<td>40.0</td>
<td>64.0</td>
<td>100.0</td>
<td>20.0</td>
<td>0.7</td>
<td>1234.0</td>
<td>drf_grid1_model_1</td>
<td>0.7564273</td>
</tr>
<tr class="odd">
<td></td>
<td>0.6</td>
<td>bernoulli</td>
<td>25.0</td>
<td>1.0</td>
<td>-1.0</td>
<td>50.0</td>
<td>64.0</td>
<td>100.0</td>
<td>15.0</td>
<td>0.7</td>
<td>1234.0</td>
<td>drf_grid1_model_22</td>
<td>0.7440692</td>
</tr>
<tr class="even">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>25.0</td>
<td>1.0</td>
<td>3.0</td>
<td>20.0</td>
<td>16.0</td>
<td>200.0</td>
<td>35.0</td>
<td>0.7</td>
<td>1234.0</td>
<td>drf_grid1_model_14</td>
<td>0.7432266</td>
</tr>
<tr class="odd">
<td></td>
<td>0.7</td>
<td>bernoulli</td>
<td>25.0</td>
<td>1.0</td>
<td>1.0</td>
<td>50.0</td>
<td>16.0</td>
<td>50.0</td>
<td>30.0</td>
<td>0.9</td>
<td>1234.0</td>
<td>drf_grid1_model_41</td>
<td>0.7215771</td>
</tr>
</tbody>
</table>

</div>
<pre style="font-size: smaller; margin-bottom: 1em;">[100 rows x 14 columns]</pre>
</div>
</div>
</section>
<section id="compute-training-cross-validation-and-validation-aucs" class="level2">
<h2 class="anchored" data-anchor-id="compute-training-cross-validation-and-validation-aucs">Compute Training, Cross Validation, and Validation AUC’s</h2>
<div id="f4addf98-cd32-431f-9b78-aa86ff5673ad" class="cell" data-execution_count="30">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb22"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb22-1"><a href="#cb22-1" aria-hidden="true" tabindex="-1"></a>results <span class="op">=</span> []</span>
<span id="cb22-2"><a href="#cb22-2" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb22-3"><a href="#cb22-3" aria-hidden="true" tabindex="-1"></a><span class="co"># Iterate through each model in the leaderboard</span></span>
<span id="cb22-4"><a href="#cb22-4" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> model <span class="kw">in</span> leaderboard.models:</span>
<span id="cb22-5"><a href="#cb22-5" aria-hidden="true" tabindex="-1"></a>    model_id <span class="op">=</span> model.model_id                       <span class="co"># Get the model ID</span></span>
<span id="cb22-6"><a href="#cb22-6" aria-hidden="true" tabindex="-1"></a>       </span>
<span id="cb22-7"><a href="#cb22-7" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Retrieve the model from the H2O cluster</span></span>
<span id="cb22-8"><a href="#cb22-8" aria-hidden="true" tabindex="-1"></a>    model_obj <span class="op">=</span> h2o.get_model(model_id)</span>
<span id="cb22-9"><a href="#cb22-9" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb22-10"><a href="#cb22-10" aria-hidden="true" tabindex="-1"></a>    <span class="co">#validation_auc = model.auc(valid=True)          # Get the validation AUC</span></span>
<span id="cb22-11"><a href="#cb22-11" aria-hidden="true" tabindex="-1"></a>    validation_auc <span class="op">=</span> model_obj.model_performance(df_test).auc()</span>
<span id="cb22-12"><a href="#cb22-12" aria-hidden="true" tabindex="-1"></a>    </span>
<span id="cb22-13"><a href="#cb22-13" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Train AUC</span></span>
<span id="cb22-14"><a href="#cb22-14" aria-hidden="true" tabindex="-1"></a>    <span class="co">#train_auc = model_obj.auc(train=True)</span></span>
<span id="cb22-15"><a href="#cb22-15" aria-hidden="true" tabindex="-1"></a>    train_auc <span class="op">=</span> model_obj.model_performance(df_train).auc()</span>
<span id="cb22-16"><a href="#cb22-16" aria-hidden="true" tabindex="-1"></a>    </span>
<span id="cb22-17"><a href="#cb22-17" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Check for cross-validation metrics summary</span></span>
<span id="cb22-18"><a href="#cb22-18" aria-hidden="true" tabindex="-1"></a>    <span class="cf">if</span> model_obj.cross_validation_metrics_summary():</span>
<span id="cb22-19"><a href="#cb22-19" aria-hidden="true" tabindex="-1"></a>        <span class="co"># Extract AUC from the cross-validation summary</span></span>
<span id="cb22-20"><a href="#cb22-20" aria-hidden="true" tabindex="-1"></a>        cv_auc <span class="op">=</span> model_obj.cross_validation_metrics_summary().as_data_frame().loc[<span class="dv">2</span>, <span class="st">'mean'</span>]  <span class="co"># Extract mean AUC</span></span>
<span id="cb22-21"><a href="#cb22-21" aria-hidden="true" tabindex="-1"></a>    <span class="cf">else</span>:</span>
<span id="cb22-22"><a href="#cb22-22" aria-hidden="true" tabindex="-1"></a>        cv_auc <span class="op">=</span> <span class="va">None</span>  <span class="co"># If no cross-validation is available</span></span>
<span id="cb22-23"><a href="#cb22-23" aria-hidden="true" tabindex="-1"></a>    </span>
<span id="cb22-24"><a href="#cb22-24" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Append the results to the list</span></span>
<span id="cb22-25"><a href="#cb22-25" aria-hidden="true" tabindex="-1"></a>    results.append({</span>
<span id="cb22-26"><a href="#cb22-26" aria-hidden="true" tabindex="-1"></a>        <span class="st">'model_id'</span>: model_id,</span>
<span id="cb22-27"><a href="#cb22-27" aria-hidden="true" tabindex="-1"></a>        <span class="st">'train_auc'</span>: train_auc,</span>
<span id="cb22-28"><a href="#cb22-28" aria-hidden="true" tabindex="-1"></a>        <span class="st">'validation_auc'</span>: validation_auc,</span>
<span id="cb22-29"><a href="#cb22-29" aria-hidden="true" tabindex="-1"></a>        <span class="st">'cross_validation_auc'</span>: cv_auc</span>
<span id="cb22-30"><a href="#cb22-30" aria-hidden="true" tabindex="-1"></a>    })</span>
<span id="cb22-31"><a href="#cb22-31" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb22-32"><a href="#cb22-32" aria-hidden="true" tabindex="-1"></a><span class="co"># Convert the results list into a Pandas DataFrame</span></span>
<span id="cb22-33"><a href="#cb22-33" aria-hidden="true" tabindex="-1"></a>df <span class="op">=</span> pd.DataFrame(results)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
</div>
</section>
<section id="display-the-model-results-auc-performance" class="level2">
<h2 class="anchored" data-anchor-id="display-the-model-results-auc-performance">Display the model results (AUC Performance)</h2>
<div id="b8a29972-913f-4b6f-8bae-54b6914df50a" class="cell" data-scrolled="true" data-execution_count="35">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb23"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb23-1"><a href="#cb23-1" aria-hidden="true" tabindex="-1"></a>df[<span class="st">'overtraining_metric'</span>] <span class="op">=</span> np.array(df[<span class="st">'train_auc'</span>]) <span class="op">-</span> np.array(df[<span class="st">'validation_auc'</span>]) <span class="co">## the smaller the better</span></span>
<span id="cb23-2"><a href="#cb23-2" aria-hidden="true" tabindex="-1"></a>df.sort_values(<span class="st">"overtraining_metric"</span>, ascending <span class="op">=</span> <span class="va">True</span>)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-display" data-execution_count="35">
<div>


<table class="dataframe caption-top table table-sm table-striped small" data-quarto-postprocess="true" data-border="1">
<thead>
<tr class="header">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">model_id</th>
<th data-quarto-table-cell-role="th">train_auc</th>
<th data-quarto-table-cell-role="th">validation_auc</th>
<th data-quarto-table-cell-role="th">cross_validation_auc</th>
<th data-quarto-table-cell-role="th">overtraining_metric</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td data-quarto-table-cell-role="th">31</td>
<td>drf_grid1_model_28</td>
<td>0.9405</td>
<td>0.9409</td>
<td>0.8784</td>
<td>-0.0003594</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">51</td>
<td>drf_grid1_model_23</td>
<td>0.9262</td>
<td>0.9258</td>
<td>0.8656</td>
<td>0.0003516</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">17</td>
<td>drf_grid1_model_94</td>
<td>0.9453</td>
<td>0.9419</td>
<td>0.8859</td>
<td>0.003448</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">14</td>
<td>drf_grid1_model_72</td>
<td>0.9456</td>
<td>0.9422</td>
<td>0.8882</td>
<td>0.003467</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">12</td>
<td>drf_grid1_model_49</td>
<td>0.9461</td>
<td>0.9417</td>
<td>0.8882</td>
<td>0.004408</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">39</td>
<td>drf_grid1_model_35</td>
<td>0.9357</td>
<td>0.9309</td>
<td>0.8735</td>
<td>0.004847</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">24</td>
<td>drf_grid1_model_84</td>
<td>0.9447</td>
<td>0.9394</td>
<td>0.8834</td>
<td>0.00534</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">47</td>
<td>drf_grid1_model_42</td>
<td>0.9245</td>
<td>0.9187</td>
<td>0.8683</td>
<td>0.005756</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">71</td>
<td>drf_grid1_model_15</td>
<td>0.905</td>
<td>0.8979</td>
<td>0.8516</td>
<td>0.007063</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">22</td>
<td>drf_grid1_model_12</td>
<td>0.9312</td>
<td>0.9234</td>
<td>0.882</td>
<td>0.007748</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">52</td>
<td>drf_grid1_model_50</td>
<td>0.9278</td>
<td>0.9201</td>
<td>0.8655</td>
<td>0.007752</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">23</td>
<td>drf_grid1_model_26</td>
<td>0.9371</td>
<td>0.9292</td>
<td>0.8812</td>
<td>0.007953</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">28</td>
<td>drf_grid1_model_46</td>
<td>0.934</td>
<td>0.9235</td>
<td>0.8788</td>
<td>0.01056</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">80</td>
<td>drf_grid1_model_96</td>
<td>0.8998</td>
<td>0.889</td>
<td>0.8366</td>
<td>0.0108</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">13</td>
<td>drf_grid1_model_43</td>
<td>0.9482</td>
<td>0.9366</td>
<td>0.8864</td>
<td>0.01165</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">84</td>
<td>drf_grid1_model_3</td>
<td>0.8994</td>
<td>0.8869</td>
<td>0.8313</td>
<td>0.01255</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">49</td>
<td>drf_grid1_model_45</td>
<td>0.9388</td>
<td>0.9253</td>
<td>0.8679</td>
<td>0.01355</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">44</td>
<td>drf_grid1_model_20</td>
<td>0.9395</td>
<td>0.9248</td>
<td>0.8713</td>
<td>0.01473</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">40</td>
<td>drf_grid1_model_66</td>
<td>0.9271</td>
<td>0.9123</td>
<td>0.8731</td>
<td>0.01482</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">75</td>
<td>drf_grid1_model_33</td>
<td>0.9393</td>
<td>0.912</td>
<td>0.8461</td>
<td>0.02729</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">69</td>
<td>drf_grid1_model_8</td>
<td>0.9321</td>
<td>0.9031</td>
<td>0.8526</td>
<td>0.02899</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">8</td>
<td>drf_grid1_model_86</td>
<td>0.9714</td>
<td>0.9414</td>
<td>0.8932</td>
<td>0.02999</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">72</td>
<td>drf_grid1_model_99</td>
<td>0.9326</td>
<td>0.9012</td>
<td>0.8491</td>
<td>0.03143</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">76</td>
<td>drf_grid1_model_76</td>
<td>0.9307</td>
<td>0.8972</td>
<td>0.8387</td>
<td>0.03353</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">7</td>
<td>drf_grid1_model_73</td>
<td>0.9824</td>
<td>0.9455</td>
<td>0.8931</td>
<td>0.03691</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">25</td>
<td>drf_grid1_model_24</td>
<td>0.9716</td>
<td>0.9293</td>
<td>0.8796</td>
<td>0.04237</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">59</td>
<td>drf_grid1_model_39</td>
<td>0.9682</td>
<td>0.9257</td>
<td>0.86</td>
<td>0.04248</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">10</td>
<td>drf_grid1_model_37</td>
<td>0.982</td>
<td>0.9393</td>
<td>0.8895</td>
<td>0.04267</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">41</td>
<td>drf_grid1_model_56</td>
<td>0.9805</td>
<td>0.9377</td>
<td>0.8731</td>
<td>0.04276</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">35</td>
<td>drf_grid1_model_11</td>
<td>0.9801</td>
<td>0.9368</td>
<td>0.8781</td>
<td>0.04329</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">16</td>
<td>drf_grid1_model_18</td>
<td>0.9823</td>
<td>0.9367</td>
<td>0.8851</td>
<td>0.04562</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">29</td>
<td>drf_grid1_model_95</td>
<td>0.9825</td>
<td>0.9364</td>
<td>0.8796</td>
<td>0.04607</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">33</td>
<td>drf_grid1_model_91</td>
<td>0.9791</td>
<td>0.9312</td>
<td>0.8787</td>
<td>0.04788</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">3</td>
<td>drf_grid1_model_79</td>
<td>0.9992</td>
<td>0.9504</td>
<td>0.9014</td>
<td>0.04877</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">53</td>
<td>drf_grid1_model_83</td>
<td>0.9751</td>
<td>0.9262</td>
<td>0.8639</td>
<td>0.04892</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">30</td>
<td>drf_grid1_model_48</td>
<td>1.0</td>
<td>0.9491</td>
<td>0.8779</td>
<td>0.05093</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">73</td>
<td>drf_grid1_model_2</td>
<td>0.9739</td>
<td>0.9224</td>
<td>0.8482</td>
<td>0.05146</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">48</td>
<td>drf_grid1_model_100</td>
<td>0.9711</td>
<td>0.9194</td>
<td>0.8669</td>
<td>0.05167</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">2</td>
<td>drf_grid1_model_25</td>
<td>0.9973</td>
<td>0.9432</td>
<td>0.9038</td>
<td>0.05406</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">4</td>
<td>drf_grid1_model_52</td>
<td>0.9949</td>
<td>0.9405</td>
<td>0.8993</td>
<td>0.05443</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">19</td>
<td>drf_grid1_model_9</td>
<td>0.9999</td>
<td>0.9439</td>
<td>0.8827</td>
<td>0.05598</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">81</td>
<td>drf_grid1_model_38</td>
<td>0.9318</td>
<td>0.8747</td>
<td>0.8331</td>
<td>0.05705</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">6</td>
<td>drf_grid1_model_64</td>
<td>0.9987</td>
<td>0.941</td>
<td>0.892</td>
<td>0.0577</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">21</td>
<td>drf_grid1_model_36</td>
<td>0.9998</td>
<td>0.9414</td>
<td>0.8819</td>
<td>0.05839</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">1</td>
<td>drf_grid1_model_85</td>
<td>0.9985</td>
<td>0.9384</td>
<td>0.9034</td>
<td>0.0601</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">56</td>
<td>drf_grid1_model_90</td>
<td>0.9878</td>
<td>0.9275</td>
<td>0.8622</td>
<td>0.06024</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">9</td>
<td>drf_grid1_model_21</td>
<td>0.995</td>
<td>0.9342</td>
<td>0.8918</td>
<td>0.06074</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">65</td>
<td>drf_grid1_model_75</td>
<td>0.9801</td>
<td>0.9176</td>
<td>0.8553</td>
<td>0.06248</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">11</td>
<td>drf_grid1_model_82</td>
<td>0.9971</td>
<td>0.9341</td>
<td>0.8879</td>
<td>0.06301</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">58</td>
<td>drf_grid1_model_54</td>
<td>1.0</td>
<td>0.9358</td>
<td>0.8569</td>
<td>0.06422</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">34</td>
<td>drf_grid1_model_4</td>
<td>0.9999</td>
<td>0.9351</td>
<td>0.8767</td>
<td>0.06476</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">60</td>
<td>drf_grid1_model_69</td>
<td>1.0</td>
<td>0.9341</td>
<td>0.8592</td>
<td>0.06591</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">0</td>
<td>drf_grid1_model_78</td>
<td>0.9992</td>
<td>0.9331</td>
<td>0.904</td>
<td>0.0661</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">5</td>
<td>drf_grid1_model_81</td>
<td>0.9955</td>
<td>0.9263</td>
<td>0.8981</td>
<td>0.06918</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">37</td>
<td>drf_grid1_model_40</td>
<td>0.9976</td>
<td>0.9281</td>
<td>0.8746</td>
<td>0.06959</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">67</td>
<td>drf_grid1_model_92</td>
<td>0.9761</td>
<td>0.9061</td>
<td>0.8537</td>
<td>0.06993</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">18</td>
<td>drf_grid1_model_98</td>
<td>0.9936</td>
<td>0.9232</td>
<td>0.8824</td>
<td>0.0704</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">20</td>
<td>drf_grid1_model_87</td>
<td>0.9982</td>
<td>0.9277</td>
<td>0.8825</td>
<td>0.07042</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">38</td>
<td>drf_grid1_model_19</td>
<td>0.9999</td>
<td>0.9287</td>
<td>0.8725</td>
<td>0.07122</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">57</td>
<td>drf_grid1_model_89</td>
<td>0.9995</td>
<td>0.9276</td>
<td>0.862</td>
<td>0.07181</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">36</td>
<td>drf_grid1_model_67</td>
<td>0.995</td>
<td>0.9224</td>
<td>0.8769</td>
<td>0.07263</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">27</td>
<td>drf_grid1_model_17</td>
<td>0.9997</td>
<td>0.927</td>
<td>0.8796</td>
<td>0.07271</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">15</td>
<td>drf_grid1_model_53</td>
<td>0.9999</td>
<td>0.9263</td>
<td>0.8852</td>
<td>0.07359</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">46</td>
<td>drf_grid1_model_5</td>
<td>0.9979</td>
<td>0.9233</td>
<td>0.8661</td>
<td>0.07462</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">50</td>
<td>drf_grid1_model_47</td>
<td>0.9931</td>
<td>0.9179</td>
<td>0.8653</td>
<td>0.07528</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">32</td>
<td>drf_grid1_model_97</td>
<td>0.9999</td>
<td>0.9239</td>
<td>0.8797</td>
<td>0.07602</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">66</td>
<td>drf_grid1_model_60</td>
<td>0.9882</td>
<td>0.9112</td>
<td>0.8548</td>
<td>0.07696</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">26</td>
<td>drf_grid1_model_61</td>
<td>0.9979</td>
<td>0.9204</td>
<td>0.8802</td>
<td>0.07755</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">54</td>
<td>drf_grid1_model_62</td>
<td>0.9976</td>
<td>0.9191</td>
<td>0.8627</td>
<td>0.07842</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">70</td>
<td>drf_grid1_model_7</td>
<td>0.9999</td>
<td>0.921</td>
<td>0.8492</td>
<td>0.07888</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">42</td>
<td>drf_grid1_model_59</td>
<td>0.9995</td>
<td>0.9196</td>
<td>0.8712</td>
<td>0.07984</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">61</td>
<td>drf_grid1_model_70</td>
<td>0.9983</td>
<td>0.9178</td>
<td>0.8603</td>
<td>0.08048</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">43</td>
<td>drf_grid1_model_29</td>
<td>0.9998</td>
<td>0.9177</td>
<td>0.8715</td>
<td>0.08217</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">82</td>
<td>drf_grid1_model_80</td>
<td>0.9995</td>
<td>0.9168</td>
<td>0.8294</td>
<td>0.08273</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">78</td>
<td>drf_grid1_model_65</td>
<td>1.0</td>
<td>0.9129</td>
<td>0.8372</td>
<td>0.08713</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">62</td>
<td>drf_grid1_model_55</td>
<td>1.0</td>
<td>0.9118</td>
<td>0.8579</td>
<td>0.08823</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">74</td>
<td>drf_grid1_model_34</td>
<td>1.0</td>
<td>0.9116</td>
<td>0.8436</td>
<td>0.08834</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">45</td>
<td>drf_grid1_model_16</td>
<td>0.9992</td>
<td>0.9101</td>
<td>0.8685</td>
<td>0.08912</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">63</td>
<td>drf_grid1_model_58</td>
<td>0.9956</td>
<td>0.9056</td>
<td>0.8577</td>
<td>0.08999</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">55</td>
<td>drf_grid1_model_32</td>
<td>0.9999</td>
<td>0.9075</td>
<td>0.8632</td>
<td>0.09248</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">83</td>
<td>drf_grid1_model_74</td>
<td>0.9971</td>
<td>0.9005</td>
<td>0.8337</td>
<td>0.09664</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">68</td>
<td>drf_grid1_model_63</td>
<td>0.9979</td>
<td>0.8969</td>
<td>0.8502</td>
<td>0.1011</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">77</td>
<td>drf_grid1_model_68</td>
<td>0.9978</td>
<td>0.8954</td>
<td>0.8414</td>
<td>0.1024</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">89</td>
<td>drf_grid1_model_31</td>
<td>1.0</td>
<td>0.896</td>
<td>0.8077</td>
<td>0.104</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">64</td>
<td>drf_grid1_model_10</td>
<td>1.0</td>
<td>0.8944</td>
<td>0.8562</td>
<td>0.1056</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">88</td>
<td>drf_grid1_model_77</td>
<td>1.0</td>
<td>0.8941</td>
<td>0.8138</td>
<td>0.1059</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">79</td>
<td>drf_grid1_model_30</td>
<td>1.0</td>
<td>0.89</td>
<td>0.8366</td>
<td>0.11</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">86</td>
<td>drf_grid1_model_71</td>
<td>0.9988</td>
<td>0.8703</td>
<td>0.8136</td>
<td>0.1285</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">92</td>
<td>drf_grid1_model_57</td>
<td>1.0</td>
<td>0.8675</td>
<td>0.7835</td>
<td>0.1325</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">87</td>
<td>drf_grid1_model_6</td>
<td>0.9989</td>
<td>0.8652</td>
<td>0.8137</td>
<td>0.1337</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">90</td>
<td>drf_grid1_model_51</td>
<td>1.0</td>
<td>0.8631</td>
<td>0.8009</td>
<td>0.1369</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">85</td>
<td>drf_grid1_model_27</td>
<td>0.9932</td>
<td>0.8483</td>
<td>0.8169</td>
<td>0.1449</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">95</td>
<td>drf_grid1_model_44</td>
<td>1.0</td>
<td>0.8551</td>
<td>0.764</td>
<td>0.1449</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">94</td>
<td>drf_grid1_model_88</td>
<td>1.0</td>
<td>0.8545</td>
<td>0.7732</td>
<td>0.1455</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">91</td>
<td>drf_grid1_model_93</td>
<td>0.9999</td>
<td>0.8374</td>
<td>0.7914</td>
<td>0.1625</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">93</td>
<td>drf_grid1_model_13</td>
<td>1.0</td>
<td>0.8319</td>
<td>0.7805</td>
<td>0.1681</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">96</td>
<td>drf_grid1_model_1</td>
<td>1.0</td>
<td>0.8157</td>
<td>0.7509</td>
<td>0.1843</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">97</td>
<td>drf_grid1_model_22</td>
<td>1.0</td>
<td>0.8049</td>
<td>0.7404</td>
<td>0.1951</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">98</td>
<td>drf_grid1_model_14</td>
<td>1.0</td>
<td>0.7898</td>
<td>0.7366</td>
<td>0.2102</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">99</td>
<td>drf_grid1_model_41</td>
<td>0.9998</td>
<td>0.7895</td>
<td>0.7207</td>
<td>0.2103</td>
</tr>
</tbody>
</table>

</div>
</div>
</div>
</section>
<section id="save-grid" class="level2">
<h2 class="anchored" data-anchor-id="save-grid">Save Grid</h2>
<div id="e89c128b-8f98-4746-b80a-c8744f4c29d8" class="cell" data-execution_count="37">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb24"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb24-1"><a href="#cb24-1" aria-hidden="true" tabindex="-1"></a>h2o.save_grid(<span class="st">"giggle_w2flag_models_grid"</span>, rf_grid.grid_id)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-display" data-execution_count="37">
<pre><code>'giggle_w2flag_models_grid/drf_grid1'</code></pre>
</div>
</div>
<div id="f799c478-a6ee-4b71-a313-b71be7125876" class="cell">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb26"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb26-1"><a href="#cb26-1" aria-hidden="true" tabindex="-1"></a><span class="co">## Best Model 'drf_grid1_model_15'</span></span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
</div>
</section>
<section id="get-h2o-model" class="level2">
<h2 class="anchored" data-anchor-id="get-h2o-model">Get H2O Model</h2>
<div id="646bf797-0234-497f-a2f3-452da9424ff2" class="cell" data-execution_count="40">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb27"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb27-1"><a href="#cb27-1" aria-hidden="true" tabindex="-1"></a>model <span class="op">=</span> h2o.get_model(<span class="st">'drf_grid1_model_12'</span>)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
</div>
</section>
<section id="variable-importance" class="level2">
<h2 class="anchored" data-anchor-id="variable-importance">Variable Importance</h2>
<div id="3a84bcad-16dd-4a7b-bb5a-7d087611357f" class="cell" data-scrolled="true" data-execution_count="57">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb28"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb28-1"><a href="#cb28-1" aria-hidden="true" tabindex="-1"></a><span class="co">#Get variable importance as a Pandas DataFrame</span></span>
<span id="cb28-2"><a href="#cb28-2" aria-hidden="true" tabindex="-1"></a>var_importance <span class="op">=</span> model.varimp(use_pandas<span class="op">=</span><span class="va">True</span>)</span>
<span id="cb28-3"><a href="#cb28-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb28-4"><a href="#cb28-4" aria-hidden="true" tabindex="-1"></a><span class="co"># Sort by scaled importance for better visualization</span></span>
<span id="cb28-5"><a href="#cb28-5" aria-hidden="true" tabindex="-1"></a>var_importance <span class="op">=</span> var_importance.sort_values(<span class="st">'scaled_importance'</span>, ascending<span class="op">=</span><span class="va">True</span>)</span>
<span id="cb28-6"><a href="#cb28-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb28-7"><a href="#cb28-7" aria-hidden="true" tabindex="-1"></a><span class="co"># Plot variable importance</span></span>
<span id="cb28-8"><a href="#cb28-8" aria-hidden="true" tabindex="-1"></a>plt.figure(figsize<span class="op">=</span>(<span class="dv">14</span>, <span class="dv">18</span>))</span>
<span id="cb28-9"><a href="#cb28-9" aria-hidden="true" tabindex="-1"></a>plt.barh(var_importance[<span class="st">'variable'</span>], var_importance[<span class="st">'scaled_importance'</span>], color<span class="op">=</span><span class="st">'skyblue'</span>)</span>
<span id="cb28-10"><a href="#cb28-10" aria-hidden="true" tabindex="-1"></a>plt.xlabel(<span class="st">'Scaled Importance'</span>)</span>
<span id="cb28-11"><a href="#cb28-11" aria-hidden="true" tabindex="-1"></a>plt.ylabel(<span class="st">'Features'</span>)</span>
<span id="cb28-12"><a href="#cb28-12" aria-hidden="true" tabindex="-1"></a>plt.title(<span class="st">'Variable Importance Plot for H2O Model'</span>)</span>
<span id="cb28-13"><a href="#cb28-13" aria-hidden="true" tabindex="-1"></a>plt.tight_layout()</span>
<span id="cb28-14"><a href="#cb28-14" aria-hidden="true" tabindex="-1"></a>plt.show()</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-21-output-1.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
</div>
</section>
<section id="test-auc-performance" class="level2">
<h2 class="anchored" data-anchor-id="test-auc-performance">Test AUC Performance</h2>
<div id="c94206e4-5831-4501-863b-f3db82e9dd50" class="cell" data-execution_count="66">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb29"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb29-1"><a href="#cb29-1" aria-hidden="true" tabindex="-1"></a>scores <span class="op">=</span> model.predict(df_test)</span>
<span id="cb29-2"><a href="#cb29-2" aria-hidden="true" tabindex="-1"></a>test <span class="op">=</span> df_test.as_data_frame()</span>
<span id="cb29-3"><a href="#cb29-3" aria-hidden="true" tabindex="-1"></a>test[<span class="st">'w2_score'</span>] <span class="op">=</span> scores.as_data_frame()[<span class="st">'p1'</span>]</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-stdout">
<pre><code>drf prediction progress: |███████████████████████████████████████████████████████| (done) 100%</code></pre>
</div>
</div>
</section>
<section id="plot-roc-curve" class="level2">
<h2 class="anchored" data-anchor-id="plot-roc-curve">Plot ROC Curve</h2>
<div id="55f8558d-569e-470c-a3db-f82f03faac65" class="cell" data-execution_count="68">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb31"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb31-1"><a href="#cb31-1" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> matplotlib.pyplot <span class="im">as</span> plt</span>
<span id="cb31-2"><a href="#cb31-2" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.metrics <span class="im">import</span> roc_curve, roc_auc_score</span>
<span id="cb31-3"><a href="#cb31-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb31-4"><a href="#cb31-4" aria-hidden="true" tabindex="-1"></a><span class="co"># Example: Ensure you have your test DataFrame</span></span>
<span id="cb31-5"><a href="#cb31-5" aria-hidden="true" tabindex="-1"></a><span class="co"># test = ... # Your DataFrame with columns "w2_score" and "w2_outcome"</span></span>
<span id="cb31-6"><a href="#cb31-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb31-7"><a href="#cb31-7" aria-hidden="true" tabindex="-1"></a><span class="co"># Extract actual outcomes and predicted scores</span></span>
<span id="cb31-8"><a href="#cb31-8" aria-hidden="true" tabindex="-1"></a>y_true <span class="op">=</span> test[<span class="st">"W2_INCOME"</span>]</span>
<span id="cb31-9"><a href="#cb31-9" aria-hidden="true" tabindex="-1"></a>y_scores <span class="op">=</span> test[<span class="st">"w2_score"</span>]</span>
<span id="cb31-10"><a href="#cb31-10" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb31-11"><a href="#cb31-11" aria-hidden="true" tabindex="-1"></a><span class="co"># Compute ROC curve and AUC score</span></span>
<span id="cb31-12"><a href="#cb31-12" aria-hidden="true" tabindex="-1"></a>fpr, tpr, thresholds <span class="op">=</span> roc_curve(y_true, y_scores)</span>
<span id="cb31-13"><a href="#cb31-13" aria-hidden="true" tabindex="-1"></a>auc_score <span class="op">=</span> roc_auc_score(y_true, y_scores)</span>
<span id="cb31-14"><a href="#cb31-14" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb31-15"><a href="#cb31-15" aria-hidden="true" tabindex="-1"></a><span class="co"># Plot ROC curve</span></span>
<span id="cb31-16"><a href="#cb31-16" aria-hidden="true" tabindex="-1"></a>plt.figure(figsize<span class="op">=</span>(<span class="dv">10</span>, <span class="dv">6</span>))</span>
<span id="cb31-17"><a href="#cb31-17" aria-hidden="true" tabindex="-1"></a>plt.plot(fpr, tpr, color<span class="op">=</span><span class="st">'blue'</span>, label<span class="op">=</span><span class="ss">f'ROC Curve (AUC = </span><span class="sc">{</span>auc_score<span class="sc">:.2f}</span><span class="ss">)'</span>)</span>
<span id="cb31-18"><a href="#cb31-18" aria-hidden="true" tabindex="-1"></a>plt.plot([<span class="dv">0</span>, <span class="dv">1</span>], [<span class="dv">0</span>, <span class="dv">1</span>], color<span class="op">=</span><span class="st">'grey'</span>, linestyle<span class="op">=</span><span class="st">'--'</span>, label<span class="op">=</span><span class="st">'Random Classifier'</span>)</span>
<span id="cb31-19"><a href="#cb31-19" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb31-20"><a href="#cb31-20" aria-hidden="true" tabindex="-1"></a><span class="co"># Add labels, title, and legend</span></span>
<span id="cb31-21"><a href="#cb31-21" aria-hidden="true" tabindex="-1"></a>plt.xlabel(<span class="st">'False Positive Rate'</span>, fontsize<span class="op">=</span><span class="dv">12</span>)</span>
<span id="cb31-22"><a href="#cb31-22" aria-hidden="true" tabindex="-1"></a>plt.ylabel(<span class="st">'True Positive Rate'</span>, fontsize<span class="op">=</span><span class="dv">12</span>)</span>
<span id="cb31-23"><a href="#cb31-23" aria-hidden="true" tabindex="-1"></a>plt.title(<span class="st">'Receiver Operating Characteristic (ROC) Curve'</span>, fontsize<span class="op">=</span><span class="dv">14</span>)</span>
<span id="cb31-24"><a href="#cb31-24" aria-hidden="true" tabindex="-1"></a>plt.legend(loc<span class="op">=</span><span class="st">'lower right'</span>, fontsize<span class="op">=</span><span class="dv">12</span>)</span>
<span id="cb31-25"><a href="#cb31-25" aria-hidden="true" tabindex="-1"></a>plt.grid(alpha<span class="op">=</span><span class="fl">0.3</span>)</span>
<span id="cb31-26"><a href="#cb31-26" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb31-27"><a href="#cb31-27" aria-hidden="true" tabindex="-1"></a><span class="co"># Beautify the plot</span></span>
<span id="cb31-28"><a href="#cb31-28" aria-hidden="true" tabindex="-1"></a>plt.tight_layout()</span>
<span id="cb31-29"><a href="#cb31-29" aria-hidden="true" tabindex="-1"></a>plt.show()</span>
<span id="cb31-30"><a href="#cb31-30" aria-hidden="true" tabindex="-1"></a></span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-23-output-1.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
</div>
</section>
<section id="more-detailed-view-of-top-variables" class="level2">
<h2 class="anchored" data-anchor-id="more-detailed-view-of-top-variables">More Detailed view of Top Variables</h2>
<div id="e88215ac-5517-418e-97ad-88dbf6cbc7e4" class="cell" data-scrolled="true" data-execution_count="78">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb32"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb32-1"><a href="#cb32-1" aria-hidden="true" tabindex="-1"></a>var_importance.sort_values(<span class="st">"relative_importance"</span>, ascending <span class="op">=</span> <span class="va">False</span>)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-display" data-execution_count="78">
<div>


<table class="dataframe caption-top table table-sm table-striped small" data-quarto-postprocess="true" data-border="1">
<thead>
<tr class="header">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">variable</th>
<th data-quarto-table-cell-role="th">relative_importance</th>
<th data-quarto-table-cell-role="th">scaled_importance</th>
<th data-quarto-table-cell-role="th">percentage</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td data-quarto-table-cell-role="th">0</td>
<td>EARNED_WAGE_ACCESS_COUNT_PAST_30D</td>
<td>1.271e+04</td>
<td>1.0</td>
<td>0.1101</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">1</td>
<td>EARNED_WAGE_ACCESS_COUNT_TREND_MONTH_3</td>
<td>8.813e+03</td>
<td>0.6934</td>
<td>0.07633</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">2</td>
<td>EARNED_WAGE_ACCESS_TREND_MONTH_3</td>
<td>8.748e+03</td>
<td>0.6883</td>
<td>0.07577</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">3</td>
<td>EARNED_WAGE_ACCESS_COUNT_TREND_MONTH_1</td>
<td>7.904e+03</td>
<td>0.6219</td>
<td>0.06846</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">4</td>
<td>INCOME_NEXT_30D</td>
<td>7.07e+03</td>
<td>0.5563</td>
<td>0.06124</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">5</td>
<td>GIG_INCOME_TO_INCOME_RATIO_PAST_2Y</td>
<td>6.874e+03</td>
<td>0.5408</td>
<td>0.05953</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">6</td>
<td>EARNED_WAGE_ACCESS_COUNT_PAST_270D</td>
<td>6.247e+03</td>
<td>0.4915</td>
<td>0.05411</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">7</td>
<td>INCOME_NEXT_7D</td>
<td>5.602e+03</td>
<td>0.4408</td>
<td>0.04852</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">8</td>
<td>GIG_INCOME_TO_INCOME_RATIO_PAST_270D</td>
<td>4.431e+03</td>
<td>0.3486</td>
<td>0.03838</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">9</td>
<td>EARNED_WAGE_ACCESS_TREND_MONTH_2</td>
<td>3.8e+03</td>
<td>0.299</td>
<td>0.03292</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">10</td>
<td>EARNED_WAGE_ACCESS_TREND_MONTH_1</td>
<td>3.53e+03</td>
<td>0.2777</td>
<td>0.03057</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">11</td>
<td>GIG_INCOME_TO_INCOME_RATIO_PAST_180D</td>
<td>3.033e+03</td>
<td>0.2386</td>
<td>0.02627</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">12</td>
<td>DIRECT_DEPOSIT_COUNT_TREND_MONTH_2</td>
<td>3.018e+03</td>
<td>0.2375</td>
<td>0.02614</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">13</td>
<td>RENT_TO_INCOME_RATIO_PAST_360D</td>
<td>2.802e+03</td>
<td>0.2204</td>
<td>0.02427</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">14</td>
<td>EARNED_WAGE_ACCESS_COUNT_TREND_MONTH_2</td>
<td>2.249e+03</td>
<td>0.1769</td>
<td>0.01948</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">15</td>
<td>NUMBER_OF_DAYS_SINCE_LAST_INCOME_PAST_30D</td>
<td>1.84e+03</td>
<td>0.1447</td>
<td>0.01593</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">16</td>
<td>AVERAGE_OUTFLOWS_2D_AFTER_PAYROLL_PAST_270D</td>
<td>1.679e+03</td>
<td>0.1321</td>
<td>0.01455</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">17</td>
<td>MEDIAN_OUTFLOWS_2D_AFTER_PAYROLL_PAST_270D</td>
<td>1.418e+03</td>
<td>0.1116</td>
<td>0.01228</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">18</td>
<td>PAYROLL_TO_INCOME_RATIO_PAST_360D</td>
<td>1.389e+03</td>
<td>0.1093</td>
<td>0.01203</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">19</td>
<td>PAYROLL_TO_INCOME_RATIO_PAST_270D</td>
<td>1.333e+03</td>
<td>0.1048</td>
<td>0.01154</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">20</td>
<td>DIRECT_DEPOSIT_TREND_MONTH_3</td>
<td>1.223e+03</td>
<td>0.09619</td>
<td>0.01059</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">21</td>
<td>INCOME_SOURCES_COUNT_PAST_180D</td>
<td>1.111e+03</td>
<td>0.08743</td>
<td>0.009625</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">22</td>
<td>RENT_TO_INCOME_RATIO_PAST_90D</td>
<td>1.067e+03</td>
<td>0.08395</td>
<td>0.009242</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">23</td>
<td>DIRECT_DEPOSIT_COUNT_TREND_MONTH_3</td>
<td>870.0</td>
<td>0.06845</td>
<td>0.007535</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">24</td>
<td>PAYROLL_AMOUNT_MEDIAN_PAST_270D</td>
<td>742.1</td>
<td>0.05839</td>
<td>0.006427</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">25</td>
<td>PAYROLL_AMOUNT_MEDIAN_PAST_2Y</td>
<td>737.1</td>
<td>0.058</td>
<td>0.006384</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">26</td>
<td>DIRECT_DEPOSIT_COUNT_TREND_MONTH_1</td>
<td>717.4</td>
<td>0.05644</td>
<td>0.006214</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">27</td>
<td>PAYROLL_AMOUNT_MEDIAN_PAST_180D</td>
<td>698.4</td>
<td>0.05495</td>
<td>0.006049</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">28</td>
<td>MEDIAN_OUTFLOWS_1D_AFTER_PAYROLL_PAST_2Y</td>
<td>681.1</td>
<td>0.05358</td>
<td>0.005899</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">29</td>
<td>PAYROLL_AMOUNT_MIN_PAST_90D</td>
<td>604.2</td>
<td>0.04754</td>
<td>0.005233</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">30</td>
<td>PAYROLL_AMOUNT_MIN_PAST_2Y</td>
<td>561.2</td>
<td>0.04415</td>
<td>0.004861</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">31</td>
<td>PAYROLL_AMOUNT_MAX_PAST_360D</td>
<td>557.7</td>
<td>0.04388</td>
<td>0.00483</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">32</td>
<td>PAYROLL_AMOUNT_MAX_PAST_270D</td>
<td>546.0</td>
<td>0.04296</td>
<td>0.004729</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">33</td>
<td>EARNED_WAGE_ACCESS_COUNT_PAST_180D</td>
<td>545.3</td>
<td>0.0429</td>
<td>0.004723</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">34</td>
<td>AVERAGE_RECURRING_EXPENDITURES_1D_AFTER_STABLE...</td>
<td>503.3</td>
<td>0.0396</td>
<td>0.004359</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">35</td>
<td>MEDIAN_BNPL_PAYMENT_PAST_30D</td>
<td>494.4</td>
<td>0.0389</td>
<td>0.004282</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">36</td>
<td>LARGEST_INCOMING_TRANSFER_TO_INCOMING_TRANSFER...</td>
<td>454.1</td>
<td>0.03573</td>
<td>0.003933</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">37</td>
<td>LARGEST_INCOMING_TRANSFER_TO_INCOMING_TRANSFER...</td>
<td>435.1</td>
<td>0.03423</td>
<td>0.003768</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">38</td>
<td>MEDIAN_BNPL_LOAN_AMOUNT_PAST_90D</td>
<td>401.1</td>
<td>0.03156</td>
<td>0.003474</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">39</td>
<td>LARGEST_INCOMING_TRANSFER_TO_INCOMING_TRANSFER...</td>
<td>358.8</td>
<td>0.02823</td>
<td>0.003107</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">40</td>
<td>LARGEST_INCOMING_TRANSFER_TO_INCOMING_TRANSFER...</td>
<td>355.1</td>
<td>0.02794</td>
<td>0.003076</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">41</td>
<td>LARGEST_INCOMING_TRANSFER_TO_INCOMING_TRANSFER...</td>
<td>348.6</td>
<td>0.02743</td>
<td>0.003019</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">42</td>
<td>DIRECT_DEPOSIT_TREND_MONTH_1</td>
<td>330.9</td>
<td>0.02603</td>
<td>0.002866</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">43</td>
<td>INCOMING_PAYMENT_APP_PROVIDERS_COUNT_TREND_MON...</td>
<td>329.9</td>
<td>0.02596</td>
<td>0.002857</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">44</td>
<td>INSURANCE_TREND_MONTH_2</td>
<td>320.8</td>
<td>0.02524</td>
<td>0.002778</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">45</td>
<td>MEDIAN_BNPL_LOAN_AMOUNT_PAST_360D</td>
<td>303.5</td>
<td>0.02388</td>
<td>0.002629</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">46</td>
<td>DEBT_PMT_STD_PAST_270D</td>
<td>296.9</td>
<td>0.02336</td>
<td>0.002572</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">47</td>
<td>NSF_FEE_TO_FEES_RATIO_PAST_270D</td>
<td>256.5</td>
<td>0.02018</td>
<td>0.002222</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">48</td>
<td>LARGEST_OUTGOING_TRANSFER_TO_OUTGOING_TRANSFER...</td>
<td>254.0</td>
<td>0.01998</td>
<td>0.0022</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">49</td>
<td>MEDIAN_BNPL_LOAN_AMOUNT_PAST_60D</td>
<td>247.6</td>
<td>0.01948</td>
<td>0.002145</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">50</td>
<td>PAYROLL_AMOUNT_MIN_PAST_30D</td>
<td>246.8</td>
<td>0.01941</td>
<td>0.002137</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">51</td>
<td>NONESSENTIAL_OUTFLOWS_TO_OUTFLOWS_RATIO_PAST_7D</td>
<td>246.4</td>
<td>0.01939</td>
<td>0.002135</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">52</td>
<td>DEBT_PMT_STD_PAST_180D</td>
<td>232.8</td>
<td>0.01831</td>
<td>0.002016</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">53</td>
<td>LARGEST_OUTGOING_TRANSFER_TO_OUTGOING_TRANSFER...</td>
<td>232.3</td>
<td>0.01828</td>
<td>0.002012</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">54</td>
<td>LARGEST_OUTGOING_TRANSFER_TO_OUTGOING_TRANSFER...</td>
<td>226.3</td>
<td>0.01781</td>
<td>0.00196</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">55</td>
<td>AVERAGE_OUTFLOWS_2D_AFTER_PAYROLL_PAST_90D</td>
<td>223.9</td>
<td>0.01762</td>
<td>0.00194</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">56</td>
<td>NONESSENTIAL_OUTFLOWS_TO_OUTFLOWS_RATIO_PAST_180D</td>
<td>216.7</td>
<td>0.01705</td>
<td>0.001877</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">57</td>
<td>LARGEST_OUTGOING_TRANSFER_TO_OUTGOING_TRANSFER...</td>
<td>212.5</td>
<td>0.01672</td>
<td>0.00184</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">58</td>
<td>EARNED_WAGE_ACCESS_COUNT_PAST_90D</td>
<td>191.9</td>
<td>0.0151</td>
<td>0.001662</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">59</td>
<td>ATM_WITHDRAWALS_TREND_MONTH_2</td>
<td>185.6</td>
<td>0.0146</td>
<td>0.001607</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">60</td>
<td>LARGEST_OUTGOING_TRANSFER_TO_OUTGOING_TRANSFER...</td>
<td>173.4</td>
<td>0.01364</td>
<td>0.001502</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">61</td>
<td>NONESSENTIAL_OUTFLOWS_TO_OUTFLOWS_RATIO_PAST_30D</td>
<td>171.4</td>
<td>0.01349</td>
<td>0.001485</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">62</td>
<td>LARGEST_OUTGOING_TRANSFER_TO_OUTGOING_TRANSFER...</td>
<td>165.3</td>
<td>0.013</td>
<td>0.001432</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">63</td>
<td>NONESSENTIAL_OUTFLOWS_TO_INCOME_RATIO_TREND_MO...</td>
<td>161.3</td>
<td>0.01269</td>
<td>0.001397</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">64</td>
<td>PAYROLL_AMOUNT_MIN_PAST_270D</td>
<td>158.1</td>
<td>0.01244</td>
<td>0.00137</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">65</td>
<td>NSF_FEE_TO_FEES_RATIO_PAST_2Y</td>
<td>154.2</td>
<td>0.01213</td>
<td>0.001335</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">66</td>
<td>CREDIT_CARD_PAYMENTS_TIMELINESS_PAST_180D</td>
<td>150.4</td>
<td>0.01183</td>
<td>0.001303</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">67</td>
<td>PAYROLL_AMOUNT_MIN_PAST_180D</td>
<td>142.1</td>
<td>0.01118</td>
<td>0.001231</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">68</td>
<td>PAYROLL_AMOUNT_MAX_PAST_90D</td>
<td>140.4</td>
<td>0.01105</td>
<td>0.001216</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">69</td>
<td>OBLIGATORY_OUTFLOWS_TO_OUTFLOWS_RATIO_PAST_90D</td>
<td>128.6</td>
<td>0.01012</td>
<td>0.001114</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">70</td>
<td>NONESSENTIAL_OUTFLOWS_TO_OUTFLOWS_RATIO_PAST_360D</td>
<td>103.9</td>
<td>0.008174</td>
<td>0.0008998</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">71</td>
<td>MEDIAN_BNPL_LOAN_AMOUNT_PAST_2Y</td>
<td>98.18</td>
<td>0.007725</td>
<td>0.0008504</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">72</td>
<td>AVERAGE_RECURRING_EXPENDITURES_3D_AFTER_STABLE...</td>
<td>97.69</td>
<td>0.007686</td>
<td>0.0008461</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">73</td>
<td>DEBT_PMT_STD_PAST_2Y</td>
<td>76.26</td>
<td>0.006</td>
<td>0.0006605</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">74</td>
<td>GAS_EXPENDITURE_COUNT_PAST_60D</td>
<td>61.58</td>
<td>0.004845</td>
<td>0.0005333</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">75</td>
<td>DEBT_PMT_STD_PAST_360D</td>
<td>61.22</td>
<td>0.004816</td>
<td>0.0005302</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">76</td>
<td>DETECTED_FINANCIAL_ACCOUNT_COUNT_PAST_30D</td>
<td>54.84</td>
<td>0.004315</td>
<td>0.000475</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">77</td>
<td>CREDIT_CARD_PAYMENTS_TIMELINESS_PAST_270D</td>
<td>49.57</td>
<td>0.0039</td>
<td>0.0004293</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">78</td>
<td>OUTGOING_PAYMENT_APP_PROVIDERS_COUNT_TREND_MON...</td>
<td>47.56</td>
<td>0.003742</td>
<td>0.0004119</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">79</td>
<td>LARGEST_OUTGOING_TRANSFER_TO_OUTGOING_TRANSFER...</td>
<td>47.31</td>
<td>0.003723</td>
<td>0.0004098</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">80</td>
<td>GIG_INCOME_TO_INCOME_RATIO_PAST_60D</td>
<td>44.28</td>
<td>0.003484</td>
<td>0.0003835</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">81</td>
<td>LARGEST_INCOMING_TRANSFER_TO_INCOMING_TRANSFER...</td>
<td>41.95</td>
<td>0.003301</td>
<td>0.0003633</td>
</tr>
<tr class="odd">
<td data-quarto-table-cell-role="th">82</td>
<td>MEDIAN_BNPL_LOAN_AMOUNT_PAST_270D</td>
<td>36.54</td>
<td>0.002875</td>
<td>0.0003165</td>
</tr>
<tr class="even">
<td data-quarto-table-cell-role="th">83</td>
<td>IS_PRIMARY_INCOME_WEEKLY</td>
<td>32.9</td>
<td>0.002589</td>
<td>0.000285</td>
</tr>
</tbody>
</table>

</div>
</div>
</div>
</section>
<section id="h2o-explainability-plots" class="level2">
<h2 class="anchored" data-anchor-id="h2o-explainability-plots">H2O Explainability Plots</h2>
<div id="1494d105-3312-4042-ba61-88de34ad0e7b" class="cell" data-execution_count="154">
<details class="code-fold">
<summary>Code</summary>
<div class="sourceCode cell-code" id="cb33"><pre class="sourceCode python code-with-copy"><code class="sourceCode python"><span id="cb33-1"><a href="#cb33-1" aria-hidden="true" tabindex="-1"></a>h2o.explain(model, df_test, columns <span class="op">=</span> [<span class="st">'EARNED_WAGE_ACCESS_COUNT_PAST_30D'</span>,</span>
<span id="cb33-2"><a href="#cb33-2" aria-hidden="true" tabindex="-1"></a>                                                             <span class="st">'EARNED_WAGE_ACCESS_COUNT_TREND_MONTH_3'</span>,</span>
<span id="cb33-3"><a href="#cb33-3" aria-hidden="true" tabindex="-1"></a>                                                             <span class="st">'EARNED_WAGE_ACCESS_TREND_MONTH_3'</span>,</span>
<span id="cb33-4"><a href="#cb33-4" aria-hidden="true" tabindex="-1"></a>                                       <span class="st">'INCOME_NEXT_30D'</span>,</span>
<span id="cb33-5"><a href="#cb33-5" aria-hidden="true" tabindex="-1"></a>                                       <span class="st">'GIG_INCOME_TO_INCOME_RATIO_PAST_60D'</span>],</span>
<span id="cb33-6"><a href="#cb33-6" aria-hidden="true" tabindex="-1"></a>           include_explanations <span class="op">=</span> [<span class="st">'confusion_matrix'</span>,</span>
<span id="cb33-7"><a href="#cb33-7" aria-hidden="true" tabindex="-1"></a>                                   <span class="st">'varimp'</span>,</span>
<span id="cb33-8"><a href="#cb33-8" aria-hidden="true" tabindex="-1"></a>                                   <span class="st">'pdp'</span>,</span>
<span id="cb33-9"><a href="#cb33-9" aria-hidden="true" tabindex="-1"></a>                                  <span class="st">'varimp_heatmap'</span>,</span>
<span id="cb33-10"><a href="#cb33-10" aria-hidden="true" tabindex="-1"></a>                                  <span class="st">'shap_summary'</span>,</span>
<span id="cb33-11"><a href="#cb33-11" aria-hidden="true" tabindex="-1"></a>                                  <span class="st">'ice'</span>,<span class="st">'model_correlation_heatmap'</span>])</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</details>
<div class="cell-output cell-output-display">
<h1>Confusion Matrix</h1>
</div>
<div class="cell-output cell-output-display">
<blockquote class="blockquote">Confusion matrix shows a predicted class vs an actual class.</blockquote>
</div>
<div class="cell-output cell-output-display">
<h2 class="anchored">drf_grid1_model_12</h2>
</div>
<div class="cell-output cell-output-display">

<style>

#h2o-table-5.h2o-container {
  overflow-x: auto;
}
#h2o-table-5 .h2o-table {
  /* width: 100%; */
  margin-top: 1em;
  margin-bottom: 1em;
}
#h2o-table-5 .h2o-table caption {
  white-space: nowrap;
  caption-side: top;
  text-align: left;
  /* margin-left: 1em; */
  margin: 0;
  font-size: larger;
}
#h2o-table-5 .h2o-table thead {
  white-space: nowrap; 
  position: sticky;
  top: 0;
  box-shadow: 0 -1px inset;
}
#h2o-table-5 .h2o-table tbody {
  overflow: auto;
}
#h2o-table-5 .h2o-table th,
#h2o-table-5 .h2o-table td {
  text-align: right;
  /* border: 1px solid; */
}
#h2o-table-5 .h2o-table tr:nth-child(even) {
  /* background: #F5F5F5 */
}

</style>      
<div id="h2o-table-5" class="h2o-container">
  
<table class="h2o-table caption-top table table-sm table-striped small" data-quarto-postprocess="true">
<caption>Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.051718291665807735</caption>
<thead>
<tr class="header">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">0</th>
<th data-quarto-table-cell-role="th">1</th>
<th data-quarto-table-cell-role="th">Error</th>
<th data-quarto-table-cell-role="th">Rate</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>0</td>
<td>9944.0</td>
<td>35.0</td>
<td>0.0035</td>
<td>(35.0/9979.0)</td>
</tr>
<tr class="even">
<td>1</td>
<td>79.0</td>
<td>24.0</td>
<td>0.767</td>
<td>(79.0/103.0)</td>
</tr>
<tr class="odd">
<td>Total</td>
<td>10023.0</td>
<td>59.0</td>
<td>0.0113</td>
<td>(114.0/10082.0)</td>
</tr>
</tbody>
</table>

</div>
</div>
<div class="cell-output cell-output-display">
<h1>Variable Importance</h1>
</div>
<div class="cell-output cell-output-display">
<blockquote class="blockquote">The variable importance plot shows the relative importance of the most important variables in the model.</blockquote>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-7.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-stdout">
<pre><code>
</code></pre>
</div>
<div class="cell-output cell-output-display">
<h1>SHAP Summary</h1>
</div>
<div class="cell-output cell-output-display">
<blockquote class="blockquote">SHAP summary plot shows the contribution of the features for each instance (row of data). The sum of the feature contributions and the bias term is equal to the raw prediction of the model, i.e., prediction before applying inverse link function.</blockquote>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-11.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-stdout">
<pre><code>
</code></pre>
</div>
<div class="cell-output cell-output-display">
<h1>Partial Dependence Plots</h1>
</div>
<div class="cell-output cell-output-display">
<blockquote class="blockquote">Partial dependence plot (PDP) gives a graphical depiction of the marginal effect of a variable on the response. The effect of a variable is measured in change in the mean response. PDP assumes independence between the feature for which is the PDP computed and the rest.</blockquote>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-15.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-stdout">
<pre><code>
</code></pre>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-17.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-stdout">
<pre><code>
</code></pre>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-19.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-stdout">
<pre><code>
</code></pre>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-21.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-stdout">
<pre><code>
</code></pre>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-23.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-stdout">
<pre><code>
</code></pre>
</div>
<div class="cell-output cell-output-display">
<h1>Confusion Matrix</h1>
</div>
<div class="cell-output cell-output-display">
<blockquote class="blockquote">Confusion matrix shows a predicted class vs an actual class.</blockquote>
</div>
<div class="cell-output cell-output-display">
<h2 class="anchored">drf_grid1_model_12</h2>
</div>
<div class="cell-output cell-output-display">

<style>

#h2o-table-6.h2o-container {
  overflow-x: auto;
}
#h2o-table-6 .h2o-table {
  /* width: 100%; */
  margin-top: 1em;
  margin-bottom: 1em;
}
#h2o-table-6 .h2o-table caption {
  white-space: nowrap;
  caption-side: top;
  text-align: left;
  /* margin-left: 1em; */
  margin: 0;
  font-size: larger;
}
#h2o-table-6 .h2o-table thead {
  white-space: nowrap; 
  position: sticky;
  top: 0;
  box-shadow: 0 -1px inset;
}
#h2o-table-6 .h2o-table tbody {
  overflow: auto;
}
#h2o-table-6 .h2o-table th,
#h2o-table-6 .h2o-table td {
  text-align: right;
  /* border: 1px solid; */
}
#h2o-table-6 .h2o-table tr:nth-child(even) {
  /* background: #F5F5F5 */
}

</style>      
<div id="h2o-table-6" class="h2o-container">
  
<table class="h2o-table caption-top table table-sm table-striped small" data-quarto-postprocess="true">
<caption>Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.051718291665807735</caption>
<thead>
<tr class="header">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">0</th>
<th data-quarto-table-cell-role="th">1</th>
<th data-quarto-table-cell-role="th">Error</th>
<th data-quarto-table-cell-role="th">Rate</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>0</td>
<td>9944.0</td>
<td>35.0</td>
<td>0.0035</td>
<td>(35.0/9979.0)</td>
</tr>
<tr class="even">
<td>1</td>
<td>79.0</td>
<td>24.0</td>
<td>0.767</td>
<td>(79.0/103.0)</td>
</tr>
<tr class="odd">
<td>Total</td>
<td>10023.0</td>
<td>59.0</td>
<td>0.0113</td>
<td>(114.0/10082.0)</td>
</tr>
</tbody>
</table>

</div>
</div>
<div class="cell-output cell-output-display">
<h1>Variable Importance</h1>
</div>
<div class="cell-output cell-output-display">
<blockquote class="blockquote">The variable importance plot shows the relative importance of the most important variables in the model.</blockquote>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-31.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-display">
<h1>SHAP Summary</h1>
</div>
<div class="cell-output cell-output-display">
<blockquote class="blockquote">SHAP summary plot shows the contribution of the features for each instance (row of data). The sum of the feature contributions and the bias term is equal to the raw prediction of the model, i.e., prediction before applying inverse link function.</blockquote>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-34.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-display">
<h1>Partial Dependence Plots</h1>
</div>
<div class="cell-output cell-output-display">
<blockquote class="blockquote">Partial dependence plot (PDP) gives a graphical depiction of the marginal effect of a variable on the response. The effect of a variable is measured in change in the mean response. PDP assumes independence between the feature for which is the PDP computed and the rest.</blockquote>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-37.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-38.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-39.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-40.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
<div class="cell-output cell-output-display">
<div>
<figure class="figure">
<p><img src="Modeling_W2_Flags-Quarto_files/figure-html/cell-25-output-41.png" class="img-fluid figure-img"></p>
</figure>
</div>
</div>
</div>
</section>

</main>
<!-- /main column -->
