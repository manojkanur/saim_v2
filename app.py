import os
import uuid
import traceback
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import os
import traceback
from flask import jsonify, current_app as app
# LangChain & LLMs
from langchain.chains import ConversationChain, RetrievalQA, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# Web search via SerpAPI
from serpapi import GoogleSearch

load_dotenv()

# ─── Flask App Setup ────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="build", static_url_path="")
CORS(app)

# ─── In-Memory Stores & Optional RAG Index ──────────────────────────────────────
conversation_chains = {}  # chat_id → ConversationChain
business_plans     = {}  # chat_id → { step:int, answers:dict }

vector_folder = "vector_index"
knowledge_vectorstore = None
if os.path.exists(f"{vector_folder}/index.faiss") and os.path.exists(f"{vector_folder}/index.pkl"):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    knowledge_vectorstore = FAISS.load_local(
        vector_folder, embeddings, allow_dangerous_deserialization=True
    )
    app.logger.info("Loaded FAISS vector store.")
else:
    app.logger.info("No FAISS index found; skipping RAG.")

# ─── API Keys & Configuration ───────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
DEEPSEARCH_API_KEY = os.getenv("DEEPSEARCH_API_KEY")
DEEPSEARCH_API_URL = os.getenv("DEEPSEARCH_API_URL")  # e.g. https://api.deepseek.com/v1/search
SERPAPI_API_KEY    = os.getenv("SERPAPI_API_KEY")

for val, name in [
    (ANTHROPIC_API_KEY,  "ANTHROPIC_API_KEY"),
    (OPENAI_API_KEY,     "OPENAI_API_KEY"),
    (DEEPSEARCH_API_KEY, "DEEPSEARCH_API_KEY"),
    (DEEPSEARCH_API_URL, "DEEPSEARCH_API_URL"),
    (SERPAPI_API_KEY,    "SERPAPI_API_KEY"),
]:
    if not val:
        raise ValueError(f"Missing {name} in .env")

openai_headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

# ─── System Prompt & Helpers ────────────────────────────────────────────────────
SYSTEM_PROMPT = """
budgeting. Your role is to guide users efficiently by gathering key details and generating actionable insights *without overwhelming them with too many questions*.

✅ *Keep the conversation smooth and efficient*  
- Start with a friendly and engaging greeting.
- Ask *only the essential questions* needed to generate a meaningful response.
- If enough details are provided, proceed without asking further.
- If information is missing, make an *educated assumption* instead of asking too many follow-ups.
- Avoid frustrating the user with unnecessary back-and-forth exchanges.

### *Guided Assistance Flow*
🟢 *Step 1: Ask the User's Business Need (One-Time Selection Only)*  
“Which of the following would you like help with today?”  
- Structuring a Company Strategy  
- Creating a Business Plan  
- Developing an Annual Budget  

🟢 *Step 2: Gather Minimal but Essential Details*  
- *If enough context is provided, proceed without asking further.*  
- *If key details are missing, ask at most 2-3 questions.*  

### *Smart Questioning Approach*
🚫 *DO NOT* ask every question in a rigid sequence.  
✅ *Instead, infer details where possible and generate insights faster.*  

Example for Business Plan:  
🔹 If the user says: "I need a business plan for a cloud kitchen in Dubai."  
✔ *CORRECT:* Proceed with generating a business plan with assumptions based on industry standards.  
❌ *WRONG:* Asking: "Who is your target audience?" "What are your marketing strategies?" "What is your revenue model?" → Too many questions!

### *Generating the Output*
✅ Once all necessary data is collected, generate a *fully computed* business strategy, plan, or budget.

- *For Company Strategy:* SWOT analysis, strategic objectives, competitive positioning.  
- *For Business Plan:* Market insights, revenue model, cost analysis, and action steps.  
- *For Annual Budget:* Profit & Loss dashboard, revenue-expense breakdown.  

✅ *Use Markdown tables for structured financial insights when necessary.*  

### *Example Table Format for Data-Driven Responses*
| *Category*       | *Details*                     |
|------------------|--------------------------------|
| *Goal*           | [User's Goal]                 |
| *Key Insights*   | [Insights Derived]            |
| *Recommendations*| [Actionable Steps]            |

| *Business Type*  | *Initial Investment (AED)*  | *Key Advantages*                   | *Key Challenges*                         |
|------------------|-----------------------------|------------------------------------|------------------------------------------|
| *Cloud Kitchen*  | 150,000 - 300,000          | Lower costs, flexible menu         | High competition, delivery-dependent     |
| *Food Truck*     | 200,000 - 400,000          | Mobility, event opportunities      | Permit process, weather-dependent        |
| *Small Café*     | 400,000 - 800,000          | Regular customers, dine-in         | High rent, staff management             |
| *Restaurant*     | 800,000 - 2,000,000+       | Full dining experience, branding   | Highest startup costs, complex ops       |

✅ *Tables should NOT be enclosed in triple backticks or formatted as code blocks.*  
✅ *Use Markdown tables only for structured financial insights, NOT for general conversations.*
- deepsearch
- websearch
- create_image
Special trigger: “15min business plan” or “5min business plan” for a guided plan flow.
"""

def is_pure_greeting(text: str) -> bool:
    return text.lower().strip() in {"hello","hi","hey","hello!","hi!","hey!"}

def strip_greeting(text: str) -> str:
    parts = text.split()
    if parts and parts[0].lower().strip(",.!?") in {"hello","hi","hey"}:
        return " ".join(parts[1:])
    return text

# ─── 15-Minute Business-Plan Questions ───────────────────────────────────────────
QUESTIONS = [
    {"key":"operational_status","text":"Existing or upcoming business? (reply 'existing' or 'upcoming')"},
    {"key":"plan_objective","text":"Purpose of this plan? (e.g. investors, banks, self-learning, etc.)"},
    {"key":"language","text":"Which language should the plan be in?"},
    {"key":"business_name","text":"What is your business name?"},
    {"key":"business_description","text":"Briefly describe your business."},
    {"key":"num_employees","text":"How many employees do you have?"},
    {"key":"product_or_service","text":"Do you offer a product or a service?"},
    {"key":"sales_channel","text":"How can customers get your offering? (online/physical/both)"},
    {"key":"geography","text":"Where do you serve your customers?"},
    {"key":"product_details","text":"Describe your product or service in detail."},
    {"key":"target_customers","text":"Who are your target customer groups?"},
    {"key":"success_drivers","text":"What are the key drivers of success for your business?"},
    {"key":"challenges","text":"What are your main challenges or weaknesses?"},
    {"key":"financials","text":"Provide a high-level budget: projected revenues & expenses."},
]

# ─── Routes & Dispatcher ─────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return app.send_static_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    msg = data.get("message", "").strip()
    mode = data.get("mode", "chat")
    chat_id = data.get("chat_id") or str(uuid.uuid4())

    if not msg:
        return jsonify({"error": "Missing 'message'"}), 400

    # 🌐 Explicit mode-based dispatch
    if mode == "business_plan":
        return business_plan_pipeline(chat_id, msg)
    elif mode == "deepsearch":
        return deep_search_pipeline(chat_id, msg)
    elif mode == "websearch":
        return web_search_pipeline(chat_id, msg)
    elif mode == "create_image":
        return image_generation_pipeline(chat_id, msg)

    # 🧹 Clean up if chat mode and leftover state from other modes
    if mode == "chat" and chat_id in business_plans:
        business_plans.pop(chat_id)

    # 🔁 Resume business plan if active (fallback if mode was missed)
    if chat_id in business_plans:
        return business_plan_pipeline(chat_id, msg)

    # 🟢 Trigger new business plan via keyword
    if "business plan" in msg.lower() and ("15min" in msg.lower() or "5min" in msg.lower()):
        return business_plan_pipeline(chat_id, msg)

    # 💬 Default Claude chat
    return chat_pipeline(chat_id, msg)


# ─── 1) Chat Pipeline ────────────────────────────────────────────────────────────
def chat_pipeline(chat_id: str, user_message: str):
    if chat_id not in conversation_chains:
        mem = ConversationBufferWindowMemory(k=15, return_messages=True)
        llm = ChatAnthropic(model_name="claude-3-5-sonnet-20241022",
                            temperature=0.7, api_key=ANTHROPIC_API_KEY)
        chain = ConversationChain(llm=llm, memory=mem, verbose=True)
        chain.memory.chat_memory.messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
        conversation_chains[chat_id] = chain
    else:
        chain = conversation_chains[chat_id]

    chain.memory.chat_memory.add_user_message(user_message)
    if is_pure_greeting(user_message):
        resp = "Hello! How can I help you today?"
        chain.memory.chat_memory.add_ai_message(resp)
        return jsonify({"chat_id": chat_id, "response": resp})

    proc = strip_greeting(user_message)

    # RAG branch
    if knowledge_vectorstore and knowledge_vectorstore.index.ntotal > 0:
        retr = knowledge_vectorstore.as_retriever(search_kwargs={"k":4})
        docs = retr.invoke(proc)
        if docs:
            llm_r = ChatAnthropic(model_name="claude-3-5-sonnet-20241022",
                                  temperature=0.7, api_key=ANTHROPIC_API_KEY)
            qa = RetrievalQA.from_chain_type(llm=llm_r, chain_type="stuff", retriever=retr)
            result = qa.invoke(proc)
            ans = result.get("result") if isinstance(result, dict) else result
            if ans and not any(p in ans.lower() for p in ["i cannot","unable","sorry"]):
                chain.memory.chat_memory.add_ai_message(ans)
                return jsonify({
                    "chat_id": chat_id,
                    "response": ans,
                    "source": "knowledge-base"
                })

    # LLM fallback using .invoke()
    ctx = chain.memory.chat_memory.messages.copy()
    if isinstance(ctx[-1], HumanMessage):
        ctx[-1] = HumanMessage(content=proc)
    else:
        ctx.append(HumanMessage(content=proc))

    try:
        out = chain.llm.invoke(ctx)
        reply = out.content
        chain.memory.chat_memory.add_ai_message(reply)
    except Exception as e:
        app.logger.error("Chat pipeline error:\n" + traceback.format_exc())
        return jsonify({"chat_id": chat_id, "error": str(e)}), 500

    return jsonify({"chat_id": chat_id, "response": reply})

_ds_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = None
if _ds_api_key:
    deepseek_client = OpenAI(
        api_key=_ds_api_key,
        base_url="https://api.deepseek.com/v1"  # DeepSeek’s OpenAI-compatible base URL
    )

def deep_search_pipeline(chat_id: str, query: str):
    """
    Perform a deep search via the DeepSeek chat endpoint and return the results.
    Requires in your .env:
      DEEPSEEK_API_KEY=<your key>
    """
    # 1) Check client initialization
    if deepseek_client is None:
        return jsonify({
            "chat_id": chat_id,
            "error": "DeepSeek API key not set. Please add DEEPSEEK_API_KEY to your environment."
        }), 500

    # 2) Query DeepSeek
    try:
        ds_resp = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a deep‐search engine. "
                        "When given a query, you return relevant document excerpts and summaries."
                    )
                },
                {"role": "user", "content": query}
            ]
        )
        # Extract the assistant’s reply
        content = ds_resp.choices[0].message.content
    except Exception as e:
        app.logger.error("Deep search pipeline error:\n" + traceback.format_exc())
        return jsonify({
            "chat_id": chat_id,
            "error": f"Deep search request failed: {e}"
        }), 500

    # 3) Return JSON response
    return jsonify({
        "chat_id": chat_id,
        "response": content,
        "source": "deepsearch"
    })

# ─── 3) Web-Search Pipeline ──────────────────────────────────────────────────────
def web_search_pipeline(chat_id: str, query: str):
    try:
        params = {"q": query, "engine": "google", "api_key": SERPAPI_API_KEY}
        search = GoogleSearch(params)
        results = search.get_dict().get("organic_results", [])[:5]

        snippets = [r.get("snippet","") for r in results]
        out = [{"title":r.get("title"), "url":r.get("link")} for r in results]

        prompt = PromptTemplate(
            input_variables=["snips","query"],
            template="""
You have these search snippets for "{query}":

{snips}

Write a concise summary.
"""
        )
        llm = ChatAnthropic(model_name="claude-3-5-sonnet-20241022",
                            temperature=0.7, api_key=ANTHROPIC_API_KEY)
        chain = LLMChain(llm=llm, prompt=prompt)
        summary = chain.run(snips="\n\n".join(snippets), query=query)

        return jsonify({
            "chat_id": chat_id,
            "response": summary,
            "results": out,
            "source": "websearch"
        })
    except Exception as e:
        app.logger.error("Web search pipeline error:\n" + traceback.format_exc())
        return jsonify({"chat_id": chat_id, "error": str(e)}), 500

# ─── 4) Image-Generation Pipeline ───────────────────────────────────────────────
def image_generation_pipeline(chat_id: str, query: str):
    try:
        img_prompt = f"using gpt-image-1 Create a polished infographic with 100% text that {query}. Include charts or visuals."
        res = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=openai_headers,
            json={"prompt": img_prompt, "n": 1, "size": "1024x1024"},
            timeout=15
        )
        res.raise_for_status()
        url = res.json()["data"][0]["url"]

        return jsonify({
            "chat_id": chat_id,
            "image_url": url,
            "prompt": img_prompt,
        })
    except Exception as e:
        app.logger.error("Image generation pipeline error:\n" + traceback.format_exc())
        return jsonify({"chat_id": chat_id, "error": str(e)}), 500

# ─── 5) 15-Min Business-Plan Pipeline ─────────────────────────────────────────────
def business_plan_pipeline(chat_id: str, user_input: str):
    try:
        state = business_plans.get(chat_id, {"step":0, "answers":{}})
        step  = state["step"]

        if step > 0:
            prev_key = QUESTIONS[step-1]["key"]
            state["answers"][prev_key] = user_input

        if step < len(QUESTIONS):
            q = QUESTIONS[step]["text"]
            state["step"] = step + 1
            business_plans[chat_id] = state
            return jsonify({"chat_id": chat_id, "response": q, "source": "business_plan"})

        ans = state["answers"]
        lines = [
            f"Generate a detailed business plan in {ans['language']} with:",
            f"- Operational status: {ans['operational_status']}",
            f"- Objective: {ans['plan_objective']}",
            f"- Business name & description: {ans['business_name']} – {ans['business_description']}",
            f"- Employees: {ans['num_employees']}",
            f"- Offering: {ans['product_or_service']} via {ans['sales_channel']} in {ans['geography']}",
            f"- Product details: {ans['product_details']}",
            f"- Target customers: {ans['target_customers']}",
            f"- Success drivers: {ans['success_drivers']}",
            f"- Challenges: {ans['challenges']}",
            f"- Financial outlook: {ans['financials']}",
        ]
        full_prompt = "\n".join(lines)

        llm = ChatAnthropic(model_name="claude-3-5-sonnet-20241022",
                            temperature=0.7, api_key=ANTHROPIC_API_KEY)
        plan = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=full_prompt)
        ]).content

        business_plans.pop(chat_id, None)
        return jsonify({"chat_id": chat_id, "response": plan, "source": "business_plan"})
    except Exception as e:
        app.logger.error("Business plan pipeline error:\n" + traceback.format_exc())
        return jsonify({"chat_id": chat_id, "error": str(e)}), 500

# ─── Run the App ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
