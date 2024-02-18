<h1>Fine tune LLM using custome dataset</h1>
<h2>Overview</h2>
<p>This project allows you to fine-tune a chat system using a custom dataset. Follow the instructions below to prepare your dataset, configure your fine-tuning specifications, install necessary dependencies, and initiate the training process.</p>
<h2>Preparing Your Dataset</h2>
<p><strong>Dataset Format:</strong> Your dataset should be in a <code>.csv</code> format with two columns: <code>human</code> and <code>assistant</code>.</p>
<ul>
    <li>The <code>human</code> column should contain questions or phrases from the human user.</li>
    <li>The <code>assistant</code> column should contain responses from the chatbot.</li>
</ul>
<p><strong>Example Dataset:</strong></p>
<table>
    <tr>
        <th>human</th>
        <th>assistant</th>
    </tr>
    <tr>
        <td>How's the weather?</td>
        <td>It's sunny and warm.</td>
    </tr>
</table>
<h2>Configuration</h2>
<p>Define your fine-tuning specifications in the <code>fine_tune_specification.toml</code> file. Please be aware that using a large batch size may result in significant delays in the execution queue. This includes:</p>
<ul>
    <li><code>data_path</code>: Path to your dataset file.</li>
    <li><code>system_message</code>: A system message in the tone you desire for your chat system.</li>
    <li><code>training_parameters</code>: Any specific training parameters you wish to adjust.</li>
    <li><code>model_specification</code>: Specify the OpenAI model you plan to use for fine-tuning.</li>
</ul>

<h2>Installation</h2>
<p>Install the necessary dependencies with the following command:</p>
<pre>
pip install -r requirements.txt
</pre>
<h2>Running the Training Process</h2>
<p>Start the training process by executing:</p>
<pre>
python run.py --config_file_path ${config_file_path}
</pre>
After initiating the training process, the response ID will be added to your original config file under the name "response id". This ID is crucial for monitoring the training and validation loss as your model fine-tunes. Please note that each time you run the script, a training job will be submitted to OpenAI, unless a response ID already exists in your config file, in this case, with command: 
<pre>
python run.py --config_file_path ${config_file_path}
</pre>
you can monitor the training and validation loss directly in the terminal.
