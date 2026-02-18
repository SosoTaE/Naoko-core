use actix_web::{post, web, App, HttpResponse, HttpServer, Responder};
use anyhow::Context;
use async_trait::async_trait;
use colored::Colorize;
use qdrant_client::qdrant::PointId;
use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, PointStruct, QueryPointsBuilder, ScoredPoint, UpsertPointsBuilder, VectorParamsBuilder};
use qdrant_client::Payload;
use qdrant_client::{Qdrant, QdrantError};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io;
use std::io::Write;
use std::sync::Arc;
use actix_web::cookie::time::Error;
use tokio::sync::mpsc;
use uuid::Uuid;
pub struct ChatGPTProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl ChatGPTProvider {
    pub fn new(api_key: String) -> Self {
        // Create a client with default headers for Auth
        let mut headers = HeaderMap::new();
        let mut auth_value = HeaderValue::from_str(&format!("Bearer {}", api_key))
            .expect("Invalid API Key format");
        auth_value.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth_value);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }
}

#[async_trait]
impl AiProvider for ChatGPTProvider {
    async fn send_chat(&self, payload: ChatRequest) -> Result<ChatResponse, Box<dyn std::error::Error>> {
        let url = format!("{}/chat/completions", self.base_url);

        // Map your generic ChatRequest to OpenAI's specific format
        // OpenAI expects "messages" just like Ollama, but "stream" is optional (defaults false)
        // and "model" must be a valid OpenAI model ID (e.g., "gpt-4o", "gpt-3.5-turbo")
        let openai_payload = serde_json::json!({
            "model": payload.model, // Ensure this string is valid (e.g., "gpt-4o")
            "messages": payload.messages,
            "stream": payload.stream
        });

        // OpenAI returns a slightly different JSON structure than Ollama.
        // We need to deserialize it into a temporary struct and map it to your ChatResponse.
        #[derive(Deserialize)]
        struct OpenAIChoice {
            message: Message,
        }
        #[derive(Deserialize)]
        struct OpenAIResponse {
            choices: Vec<OpenAIChoice>,
        }

        let res = self.client.post(url)
            .json(&openai_payload)
            .send().await?
            .json::<OpenAIResponse>().await?;

        // Extract the first choice to match your ChatResponse struct
        if let Some(choice) = res.choices.into_iter().next() {
            Ok(ChatResponse {
                message: choice.message,
            })
        } else {
            Err("OpenAI returned no choices".into())
        }
    }

    async fn get_embedding(&self, text: String) -> Result<Vec<f32>, anyhow::Error> {
        let url = format!("{}/embeddings", self.base_url);

        let payload = serde_json::json!({
            "model": "text-embedding-3-small", // Standard efficient embedding model
            "input": text
        });

        // OpenAI Embedding Response Structure
        #[derive(Deserialize)]
        struct EmbeddingData {
            embedding: Vec<f32>,
        }
        #[derive(Deserialize)]
        struct OpenAIEmbeddingResponse {
            data: Vec<EmbeddingData>,
        }

        let res = self.client.post(url)
            .json(&payload)
            .send().await?
            .json::<OpenAIEmbeddingResponse>().await?;

        if let Some(data) = res.data.into_iter().next() {
            Ok(data.embedding)
        } else {
            Err(anyhow::anyhow!("OpenAI returned no embedding data"))
        }
    }
}
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Message {
    role: String,
    content: String,
}

#[async_trait]
pub trait AiProvider {
    async fn send_chat(&self, payload: ChatRequest) -> Result<ChatResponse, Box<dyn std::error::Error>>;
    async fn get_embedding(&self, text: String) -> Result<Vec<f32>, anyhow::Error>;
}


pub struct OllamaProvider {
    client: Client,
    base_url: String, // e.g., "http://localhost:11434"
}

impl OllamaProvider {
    pub fn new(client: Client, base_url: String) -> Self {
        if base_url.is_empty() {
            panic!("URL cannot be empty");
        }
        Self { client, base_url }
    }
}

#[async_trait]
impl AiProvider for OllamaProvider {
    async fn send_chat(&self, payload: ChatRequest) -> Result<ChatResponse, Box<dyn std::error::Error>> {
        let url = format!("{}/api/chat", self.base_url);
        let res = self.client.post(url).json(&payload).send().await?
            .json::<ChatResponse>().await?;
        Ok(res)
    }

    async fn get_embedding(&self, text: String) -> Result<Vec<f32>, anyhow::Error> {
        let url = format!("{}/api/embeddings", self.base_url);
        let payload = serde_json::json!({
            "model": "nomic-embed-text",
            "prompt": text
        });

        let res = self.client.post(url).json(&payload).send().await?
            .json::<EmbedResponse>().await?;
        Ok(res.embedding)
    }
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    message: Message,
}

#[derive(Serialize)]
pub struct EmbedRequest {
    pub model: String,
    pub prompt: String,
}

#[derive(Deserialize, Debug)]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
}

pub struct History {
    data: Vec<ScoredPoint>,
}

impl History {
    pub fn new(data: Vec<ScoredPoint>) -> Self {
        Self { data }
    }

    pub fn convert_to_history(&self) -> String {
        let mut history = String::new();

        history += "<conversations>\n";

        for point in &self.data {
            let is_bot = point.payload.get("bot").unwrap().as_bool().unwrap();



            if is_bot {
                history += format!("<naoko timestamp={} score={}>{}</naoko>\n", point.payload.get("timestamp").unwrap(), point.score, point.payload.get("content").unwrap().to_string()).as_str();
            } else {
                history += format!("<user timestamp={} score={}>{}</user>\n", point.payload.get("timestamp").unwrap(), point.score,point.payload.get("content").unwrap().to_string()).as_str();
            }
        }

        history += "</conversations>";

        history
    }
}

pub struct QdrantProvider {
    client: Qdrant,
    collection_name: String,
    ollama_provider: Arc<OllamaProvider>,
}

#[derive(Deserialize, Clone)]
struct Tool {
    name: String,
    executable: String,
    tag: String,
    description: String,
    args: Option<Vec<String>>,
}

async fn handle_tool_logic(response: &str) -> anyhow::Result<()> {
    // 1. Check if the specific trigger tag exists
    let trigger = "```email_dispatch";

    if let Some(start_index) = response.find(trigger) {
        // Find the start of the JSON (after the trigger tag)
        let json_start = start_index + trigger.len();

        // Find the closing triple backticks
        if let Some(end_index) = response[json_start..].find("```") {
            let json_content = response[json_start..json_start + end_index].trim();

            println!("\n🔍 Naoko: Tool Use Detected (Emailer)");

            // 2. Write the JSON to the file Naoko-style
            std::fs::write("mail_task.json", json_content)?;

            // 3. Request User Permission (Human-in-the-loop)
            println!("Naoko is requesting to send an email. Execute email_sender? (y/N)");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;

            if input.trim().to_lowercase() == "y" {
                println!("🚀 Dispatching...");

                // 4. Run your subprocess
                let status = std::process::Command::new("./tools/email_sender")
                    .arg("mail_task.json")
                    .status()?;

                if status.success() {
                    println!("✅ Naoko: System report dispatched.");
                } else {
                    println!("❌ Naoko: The dispatch failed. Check emailer logs.");
                }
            } else {
                println!("🚫 Dispatch aborted by Operator.");
            }
        }
    }
    Ok(())
}

impl QdrantProvider {
    pub fn new(url: &str, collection_name: String, ollama_provider: Arc<OllamaProvider>) -> Result<Self, QdrantError> {
        // Use from_url to configure, then .build() to instantiate
        let client = Qdrant::from_url(url).build()?;

        Ok(Self {
            client,
            collection_name,
            ollama_provider
        })
    }
    pub async fn check_and_create(&self) -> Result<(), QdrantError> {
        if !self.client.collection_exists(&self.collection_name).await? {
            println!("Creating collection: {}", self.collection_name);
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.collection_name)
                        .vectors_config(VectorParamsBuilder::new(768, Distance::Cosine)),
                )
                .await?;
        } else {
            println!("Collection {} already exists.", self.collection_name);
        }
        Ok(())
    }

    pub async fn save(&self, content: String, bot: bool) -> anyhow::Result<()>{
        // 1. Build the Payload
        let mut h = Payload::new();
        h.insert("content", content.clone());
        h.insert("timestamp", chrono::Utc::now().to_rfc3339());
        h.insert("bot", bot);

        // 2. Generate a unique ID (Important: don't use 1!)
        let point_id: PointId = Uuid::new_v4().to_string().into();

        // 3. Get Embeddings
        // Conversion: We map the generic error from Ollama into a QdrantError string
        let embeddings = self.ollama_provider
            .get_embedding(content.clone()).await?;
        // 4. Create the Point
        let points = vec![
            PointStruct::new(point_id, embeddings, h),
        ];

        // 5. Upsert (Note: Using the dynamic collection_name from the struct)
        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points))
            .await?;

        Ok(()) // Explicitly return Ok to satisfy the Result<(), E>
    }

    pub async fn search_data(&self, content: String) -> Result<History, anyhow::Error> {
        let embedding = self.ollama_provider.get_embedding(content.clone()).await?;

        let search_result = self.client
            .query(
                QueryPointsBuilder::new(self.collection_name.clone())
                    .query(embedding)
                    .with_payload(true)
                    .score_threshold(0.2)
                    .limit(100)
            )
            .await
            .context("Failed to search data")?;

        Ok(History{data:search_result.result.clone()})
    }

}

fn generate_master_prompt() -> String {
    let prompt_template = r#"
## SYSTEM IDENTITY
You are NAOKO-CORE, an automated System Intelligence for Arch Linux.

## CRITICAL INSTRUCTION: THE RESPONSE ENVELOPE
You are the central brain of a microservices architecture. You do NOT have direct access to read files, write code, or execute commands on the local operating system.

Instead, you have exactly ONE internal tool: delegating tasks to the Go Task Scheduler API.

Every single response you generate MUST be a valid JSON object matching this exact schema:

{
  "user_message": "Text to send to the user (use null if you are working silently).",
  "tool_call": {
    "name": "dispatch_task",
    "args": {
      "command": "The exact shell command you want the Go worker pool to execute (e.g., 'cat /tmp/data.txt', 'curl -X POST ...', or 'python3 -c \"print(1+1)\"')",
      "schedule": null // Leave null for immediate execution, or provide schedule object for later
    }
  },
  "status": "complete | needs_followup | waiting_for_user"
}

If you do not need to dispatch a task, set "tool_call" to null.

## STATUS FLAG DEFINITIONS
- "complete": The task is finished. The system will deliver `user_message` to the user.
- "needs_followup": You need the system to instantly trigger you again. If you output a "tool_call", the status MUST be "needs_followup" so the system knows to wait for the Go backend to finish the job and return the terminal output to you.
- "waiting_for_user": You cannot proceed until the human replies.

## DEPENDENT TASKS (ReAct Algorithm)
If a task requires multiple steps (e.g., read a file, analyze it, then send an email):
1. Output a `dispatch_task` to read the file (e.g., `cat filename.txt`), status "needs_followup".
2. Wait for the system to inject the stdout result into your memory.
3. Output the next `dispatch_task` to send the email based on the text you just received.

## CURRENT SESSION
User: SosoTaE
OS: Arch Linux
Time: {time}
"#;

    prompt_template.replace("{time}", &chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string())
}

async fn json_orchestrator(
    response: &str,
    tools: &Vec<Tool>,
    qdrant: &QdrantProvider,
) -> anyhow::Result<Option<String>> {

    // 🧠 1. INTERNAL AGENT: MEMORY RECALL
    // Matches your "memory_recall" tool tag
    let memory_tag = "```memory_recall";
    if let Some(json_content) = extract_json(response, memory_tag) {
        println!("\n🧠 {}", "Naoko is thinking (Querying Long-Term Memory)...".magenta());

        let data: Value = serde_json::from_str(&json_content)
            .context("Failed to parse memory_recall JSON")?;

        if let Some(query) = data["query"].as_str() {
            println!("🔍 Database Query: '{}'", query.cyan());

            // Search Vector DB
            let history = qdrant.search_data(query.to_string()).await?;
            let memory_context = history.convert_to_history();

            // Return the text context to the Main Loop
            if !memory_context.is_empty() {
                println!("✅ Memory Found.");
                return Ok(Some(memory_context));
            } else {
                return Ok(Some("SYSTEM NOTE: No relevant memories found in database.".to_string()));
            }
        }
    }

    // ⚡ 2. EXTERNAL ACTIONS (Email, System Stats)
    for tool in tools {
        let tag = format!("```{}", tool.tag);
        if let Some(json_content) = extract_json(response, &tag) {
            println!("\n⚡ {}", format!("Naoko Action: [{}]", tool.name).bold().bright_red());

            let mut data: Value = serde_json::from_str(&json_content)
                .context(format!("Failed to parse JSON for {}", tool.name))?;

            // 🛡️ SECURITY INJECTION (Specific to your nested 'email_dispatch' structure)
            if tool.name == "email_dispatch" {
                // Access nested path: smtp_server -> credentials -> email
                if let Some(email) = data.pointer("/smtp_server/credentials/email").and_then(|v| v.as_str()) {

                    // Fetch Password from KWallet / Secret Service
                    if let Ok(entry) = keyring::Entry::new("naoko-nexus", email) {
                        if let Ok(password) = entry.get_password() {
                            // Inject Password into nested path
                            if let Some(creds) = data.pointer_mut("/smtp_server/credentials") {
                                creds["password"] = serde_json::json!(password);
                                println!("🔐 {}", "System: SMTP Password injected from Keyring.".green());
                            }
                        }
                    }
                }
            }

            // Execute the tool
            if user_confirmed() {
                std::fs::write("task.json", serde_json::to_string_pretty(&data)?)?;

                // If it's the 'system_stats' tool or 'email_sender'
                let status = std::process::Command::new(&tool.executable)
                    .arg("task.json")
                    .status()?;

                // For system_stats, we might want to capture output and return it to AI?
                // For now, we assume these tools print to stdout.
                return Ok(None);
            }
        }
    }

    Ok(None)
}
fn extract_json(response: &str, tag: &str) -> Option<String> {
    let start_tag = tag;
    let end_tag = "```";

    if let Some(start_pos) = response.find(start_tag) {
        let search_area = &response[start_pos + start_tag.len()..];
        if let Some(end_pos) = search_area.find(end_tag) {
            return Some(search_area[..end_pos].trim().to_string());
        }
    }
    None
}
fn user_confirmed() -> bool {
    print!("🛡️ Naoko: Proceed with execution? (y/N): ");

    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read input");

    match input.trim().to_lowercase().as_str() {
        "y" | "yes" => true,
        _ => false,
    }
}

async fn handle_dynamic_tools(response: &str, registry: &Vec<Tool>) -> anyhow::Result<()> {
    for tool in registry {
        let full_tag = format!("```{}", tool.tag);
        if let Some(start) = response.find(&full_tag) {
            let json_payload = extract_json(response, &full_tag).unwrap(); // Helper to get JSON


            // Execute the specific file defined in JSON
            let mut cmd = std::process::Command::new(&tool.executable);
            if let Some(ref args) = tool.args { cmd.args(args); }
            cmd.arg("task.json"); // Pass the data file

            std::fs::write("task.json", json_payload)?;
            cmd.status()?;

        }
    }
    Ok(())
}

// 1. Define the incoming JSON structure
// What the web endpoint receives (from your Go app, Telegram, etc.)
#[derive(Deserialize, Debug)]
pub struct TriggerPayload {
    pub source: String,
    pub user_id: String,
    pub message: String,
}

// What the Actix thread sends to the Naoko Core thread
#[derive(Debug)]
pub struct TriggerEvent {
    pub source: String,
    pub user_id: String,
    pub message: String,
}

// 2. Create the Actix Endpoint
#[post("/api/trigger")]
async fn handle_trigger(
    payload: web::Json<TriggerPayload>,
    // Extract the channel sender from Actix's shared state
    tx: web::Data<mpsc::Sender<TriggerEvent>>,
) -> impl Responder {

    let event = TriggerEvent {
        source: payload.source.clone(),
        user_id: payload.user_id.clone(),
        message: payload.message.clone(),
    };

    // Send the data to the background thread!
    match tx.send(event).await {
        Ok(_) => {
            // Instantly return 202 Accepted so the web client isn't left hanging
            HttpResponse::Accepted().json(serde_json::json!({
                "status": "success",
                "detail": "Task handed over to Naoko Core"
            }))
        }
        Err(e) => {
            eprintln!("❌ Failed to send to Core: {}", e);
            HttpResponse::InternalServerError().finish()
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct AiResponse {
    pub user_message: Option<String>,
    pub tool_call: Option<Value>,
    pub status: String,
}

async fn naoko_core_loop(mut rx: mpsc::Receiver<TriggerEvent>) -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("Key not found");
    let ollama_builder = Arc::new(OllamaProvider::new(Client::new(), "http://localhost:11434".to_string()));
    let chatgpt_builder = Arc::new(ChatGPTProvider::new(api_key));
    let client = QdrantProvider::new("http://localhost:6334", "naoko_memory".to_string(), ollama_builder.clone())?;

    let tools_json = std::fs::read_to_string("tools.json")?;

    let tools: Vec<Tool> = serde_json::from_str(&tools_json)?;

    let system_message = Message {
        role: "system".to_string(),
        content: generate_master_prompt(),
    };

    client.check_and_create().await?;

    let mut messages = vec![system_message.clone()];

    while let Some(event) = rx.recv().await {
        let initial_context = client.search_data(event.message.clone()).await?.convert_to_history();

        let full_input = format!(
            "IMMEDIATE CONTEXT:\n{}\n\n{} REQUEST:\n{}",
            initial_context, event.source.clone(), event.message.clone()
        );
        messages.push(Message { role: "user".to_string(), content: full_input });

            let payload = ChatRequest {
                model: "gpt-5.2-2025-12-11".to_string(),
                messages: messages.clone(),
                stream: false,
            };

            let resp = chatgpt_builder.send_chat(payload).await?;
            let content = resp.message.content.clone();

            match serde_json::from_str::<AiResponse>(content.as_str()) {
                Ok(ai_response) => {
                    println!("{:?}", ai_response)
                }

                Err(e) => {
                    println!("{}", e)
                }
            }

            client.save(content.clone(), true).await?;
    }

    Ok(())

}

// Struct for the POST requests
#[derive(Serialize, Debug)]
struct JobPayload {
    name: String,
    command: String,
    // Omits the schedule key entirely if it is None
    #[serde(skip_serializing_if = "Option::is_none")]
    schedule: Option<Value>,
}

// Structs for parsing the GET /history response
#[derive(Deserialize, Debug)]
struct HistoryResponse {
    success: bool,
    data: Vec<ExecutionRecord>,
}

#[derive(Deserialize, Debug)]
struct ExecutionRecord {
    #[serde(rename = "ID")]
    id: u32,
    status: String,
    output: String,
    finishedAt: Option<String>,
}



pub async fn dispatch_job(
    command: String,
    schedule: Option<Value>,
    api_key: &str,
) -> Result<String, anyhow::Error> {
    let client = Client::new();
    let is_scheduled = schedule.is_some();

    // Determine the correct route
    let url = if is_scheduled {
        "http://localhost:3000/api/create/job"
    } else {
        "http://localhost:3000/api/execute"
    };

    let job_name = format!("AI Task - {}", chrono::Local::now().format("%H:%M:%S"));

    let payload = JobPayload {
        name: job_name,
        command,
        schedule,
    };

    println!("🚀 Sending POST to {}...", url);

    let response = client
        .post(url)
        .header("X-API-Key", api_key) // Authorize using API key
        .json(&payload)
        .send()
        .await?;

    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    if status.is_success() {
        Ok(format!("SUCCESS: Job dispatched to {}. Response: {}", url, body))
    } else {
        Ok(format!("ERROR: Failed with status {}. Details: {}", status, body))
    }
}

pub async fn get_job_history(
    job_id: u32,
    api_key: &str,
) -> Result<String, anyhow::Error> {
    let client = Client::new();
    let url = format!("http://localhost:3000/api/job/{}/history", job_id);

    println!("🔍 Fetching execution history for Job ID: {}", job_id);

    let response = client
        .get(&url)
        .header("X-API-Key", api_key) // Authorize using API key
        .send()
        .await?;

    if !response.status().is_success() {
        let err_body = response.text().await.unwrap_or_default();
        return Ok(format!("ERROR: Could not fetch history. Details: {}", err_body));
    }

    // Parse the JSON into our struct
    let history: HistoryResponse = response.json().await?;

    if !history.success {
        return Ok("ERROR: Go backend reported failure in the meta response.".to_string());
    }

    if history.data.is_empty() {
        return Ok("STATUS: Job is pending or has no execution history yet.".to_string());
    }

    // Grab the most recent execution record (assuming the newest is first or last)
    // Here we just grab the first one in the array
    let latest_run = &history.data[0];

    // Format the result nicely so Naoko can read the terminal output
    let result_string = format!(
        "STATUS: {}\nFINISHED AT: {:?}\nOUTPUT:\n{}",
        latest_run.status, latest_run.finishedAt, latest_run.output
    );

    Ok(result_string)
}

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (tx, rx) = mpsc::channel::<TriggerEvent>(100);
    let app_data = web::Data::new(tx);

    tokio::spawn(async move {
        naoko_core_loop(rx).await;
    });

    HttpServer::new(move || {
        App::new().app_data(app_data.clone()).service(handle_trigger)
    })
        .bind(("0.0.0.0", 8080))?
        .run()
        .await?;


    Ok(())
}
