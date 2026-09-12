from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path

import os
import json
import uuid
from datetime import datetime, timezone

from groq import Groq


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is missing.")

client = (
    Groq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Lomi API",
    description="Lomi - AI Career Companion Backend",
    version="5.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"

if not FRONTEND_FILE.exists():
    print(
        "WARNING: frontend index.html not found:",
        FRONTEND_FILE
    )


# =========================================================
# IN-MEMORY DATA
# =========================================================

conversation_memory: Dict[
    str,
    List[Dict[str, str]]
] = {}

student_progress: Dict[
    str,
    Dict[str, Any]
] = {}

MAX_HISTORY = 12


# =========================================================
# TIME HELPER
# =========================================================

def now_iso():

    return datetime.now(
        timezone.utc
    ).isoformat()


# =========================================================
# SESSION KEY
# =========================================================

def get_session_key(
    profile: Dict[str, Any],
    current_skill: str
) -> str:

    profile_copy = dict(profile)

    profile_copy.pop(
        "timestamp",
        None
    )

    return (
        json.dumps(
            profile_copy,
            sort_keys=True,
            ensure_ascii=False
        )
        + "|"
        + current_skill
    )


# =========================================================
# MEMORY
# =========================================================

def add_to_memory(
    session_key: str,
    role: str,
    content: str
):

    if session_key not in conversation_memory:

        conversation_memory[
            session_key
        ] = []

    conversation_memory[
        session_key
    ].append(
        {
            "role": role,
            "content": content
        }
    )

    conversation_memory[
        session_key
    ] = conversation_memory[
        session_key
    ][-MAX_HISTORY:]


def get_memory(
    session_key: str
):

    return conversation_memory.get(
        session_key,
        []
    )


# =========================================================
# PROGRESS HELPERS
# =========================================================

def create_default_progress():

    return {

        "student_id": "",

        "skills": {},

        "completed_topics": [],

        "weak_topics": [],

        "missions": [],

        "quiz_scores": [],

        "quiz_history": [],

        "projects": [],

        "project_scores": [],

        "best_quiz_score": 0,

        "best_project_score": 0,

        "readiness": 0,

        "total_missions": 0,

        "completed_missions": 0,

        "total_quizzes": 0,

        "completed_quizzes": 0,

        "total_projects": 0,

        "completed_projects": 0,

        "last_skill": "",

        "last_topic": "",

        "last_action": "",

        "updated_at": now_iso()
    }


def get_student_progress(
    student_id: str
):

    if student_id not in student_progress:

        data = create_default_progress()

        data["student_id"] = student_id

        student_progress[
            student_id
        ] = data

    return student_progress[
        student_id
    ]


def calculate_readiness(
    progress: Dict[str, Any]
):

    mission_score = min(
        100,
        progress.get(
            "completed_missions",
            0
        ) * 8
    )

    quiz_score = min(
        100,
        progress.get(
            "best_quiz_score",
            0
        )
    )

    project_score = min(
        100,
        progress.get(
            "best_project_score",
            0
        )
    )

    topics = len(
        progress.get(
            "completed_topics",
            []
        )
    )

    topic_score = min(
        100,
        topics * 10
    )

    readiness = round(
        (
            mission_score
            + quiz_score
            + project_score
            + topic_score
        ) / 4
    )

    return max(
        0,
        min(
            100,
            readiness
        )
    )


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):

    message: str

    profile: Optional[
        Dict[str, Any]
    ] = None

    progress: Optional[
        Dict[str, Any]
    ] = None

    current_skill: Optional[str] = None

    session_id: Optional[str] = None


class QuizRequest(BaseModel):

    student_id: str = "default_student"

    skill: str

    topic: Optional[str] = None

    difficulty: Optional[str] = "Beginner"

    question_count: int = Field(
        default=5,
        ge=3,
        le=10
    )

    previous_score: Optional[
        float
    ] = None

    mistakes: Optional[
        List[str]
    ] = []

    completed_topics: Optional[
        List[str]
    ] = []


class QuizResultRequest(BaseModel):

    student_id: str = "default_student"

    skill: str

    topic: str

    score: float = Field(
        ge=0,
        le=100
    )

    correct: int = 0

    total: int = 5

    mistakes: Optional[
        List[str]
    ] = []

    time_taken: Optional[
        int
    ] = 0


class MissionRequest(BaseModel):

    student_id: str = "default_student"

    skill: str

    topic: Optional[str] = None

    difficulty: Optional[str] = "Beginner"

    previous_score: Optional[
        float
    ] = None

    weak_topics: Optional[
        List[str]
    ] = []

    completed_topics: Optional[
        List[str]
    ] = []


class MissionCompleteRequest(BaseModel):

    student_id: str = "default_student"

    skill: str

    topic: str

    mission_title: str

    score: float = Field(
        default=100,
        ge=0,
        le=100
    )

    time_taken: Optional[
        int
    ] = 0


class ProgressRequest(BaseModel):

    student_id: str = "default_student"

    profile: Optional[
        Dict[str, Any]
    ] = None

    progress: Optional[
        Dict[str, Any]
    ] = None


class ProjectReviewRequest(BaseModel): 
 
    student_id: str 
 
    skill: str 
 
    topic: str = "" 
 
    project: str 
 
    github_url: str = "" 
 
    live_url: str = "" 
 
    description: str = "" 
 
    requirements: list = []

@app.get("/")
def serve_frontend():

    if not FRONTEND_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail="index.html not found."
        )

    return FileResponse(
        FRONTEND_FILE
    )


@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "lomi":
            "online",

        "ai":
            "Groq",

        "model":
            GROQ_MODEL,

        "memory_sessions":
            len(
                conversation_memory
            ),

        "students":
            len(
                student_progress
            ),

        "timestamp":
            now_iso()
    }


# =========================================================
# GROQ HELPER
# =========================================================

def ask_groq(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 2500
):

    if not client:

        raise HTTPException(
            status_code=500,
            detail=(
                "GROQ_API_KEY is missing. "
                "Add it to the .env file."
            )
        )

    try:

        response = (
            client
            .chat
            .completions
            .create(

                model=GROQ_MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=temperature,

                max_completion_tokens=max_tokens,

                include_reasoning=False
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:

            raise HTTPException(
                status_code=500,
                detail="Groq returned an empty response."
            )

        return content.strip()

    except HTTPException:

        raise

    except Exception as error:

        print(
            "GROQ ERROR:",
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Groq API error: {str(error)}"
        )


# =========================================================
# GROQ JSON HELPER
# =========================================================

def ask_groq_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 2500
):

    if not client:

        raise HTTPException(
            status_code=500,
            detail=(
                "GROQ_API_KEY is missing. "
                "Add it to the .env file."
            )
        )

    try:

        response = (
            client
            .chat
            .completions
            .create(

                model=GROQ_MODEL,

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            system_prompt
                    },
                    {
                        "role":
                            "user",

                        "content":
                            user_prompt
                    }
                ],

                temperature=
                    temperature,

                max_completion_tokens=
                    max_tokens,

                include_reasoning=
                    False,

                response_format={
                    "type":
                        "json_object"
                }
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:

            raise HTTPException(
                status_code=500,
                detail=
                    "Groq returned an empty response."
            )

        try:

            return json.loads(
                content
            )

        except json.JSONDecodeError:

            print(
                "INVALID GROQ JSON:"
            )

            print(content)

            raise HTTPException(
                status_code=500,
                detail=
                    "Groq returned invalid JSON."
            )

    except HTTPException:

        raise

    except Exception as error:

        print(
            "GROQ ERROR:",
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail=
                f"Groq API error: {str(error)}"
        )


# =========================================================
# LOMI CHAT — CHATGPT STYLE AI COPILOT
# =========================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    profile = request.profile or {}
    progress = request.progress or {}

    current_skill = request.current_skill or "Not selected"

    # -----------------------------------------------------
    # SESSION
    # -----------------------------------------------------

    if request.session_id:
        session_key = request.session_id
    else:
        session_key = get_session_key(
            profile,
            current_skill
        )

    # -----------------------------------------------------
    # GET PREVIOUS CONVERSATION
    # -----------------------------------------------------

    previous_messages = get_memory(session_key)

    # -----------------------------------------------------
    # SYSTEM PROMPT
    # -----------------------------------------------------

    system_prompt = """
You are Lomi.

Lomi is a personal AI Career Companion, Study Buddy,
Learning Coach, Project Mentor and Placement Mentor.

Your personality should feel like a smart, helpful AI
companion similar to ChatGPT, but personalized for the
student's education and career journey.

=========================================================
CORE BEHAVIOR
=========================================================

Your most important rule:

ANSWER THE USER'S ACTUAL QUESTION.

Do NOT automatically create a learning plan.

Do NOT automatically create a mission.

Do NOT automatically create a quiz.

Do NOT automatically create a project.

Do NOT automatically return JSON.

Do NOT force every response into cards or sections.

Have a natural conversation.

Understand what the student is asking and give the most
useful answer for that specific question.

=========================================================
CONVERSATION
=========================================================

Remember the previous conversation.

Use previous messages to understand context.

If the student asks a follow-up question, continue from
the previous discussion instead of starting from zero.

Example:

Student:
"What is a variable in Python?"

Then:

Student:
"Give me an example."

You should understand that "example" refers to Python
variables.

=========================================================
STUDENT CONTEXT
=========================================================

Use the student's profile and progress when useful.

Possible profile information:

- Name
- Year
- Branch
- Career goal
- Target role
- Current skill

Possible progress information:

- Readiness
- Completed missions
- Quiz scores
- Project scores
- Weak topics
- Completed topics
- Skills
- Previous activities

Do not mention private/internal data unnecessarily.

Use the information to personalize the answer.

=========================================================
ANY QUESTION
=========================================================

The student can ask about ANYTHING related to:

Python
Java
C
C++
JavaScript
HTML
CSS
React
SQL
DSA
Data Science
Machine Learning
Statistics
Excel
Power BI
Tableau
Aptitude
Communication
English
Git
GitHub
Cloud
DevOps
Cyber Security
AI
Generative AI
Resume
Projects
Internships
Placements
Interviews
Career planning
College learning
Study planning

The student may also ask normal conversational questions.

Answer naturally.

=========================================================
WHEN EXPLAINING
=========================================================

Explain according to the student's level.

If the student is a beginner:

- Use simple language.
- Use real-life examples.
- Explain step by step.
- Avoid unnecessary advanced terminology.
- Give examples when useful.

If the student already understands the basics:

- Go deeper.
- Give practical examples.
- Point out common mistakes.
- Increase difficulty gradually.

=========================================================
WHEN THE STUDENT ASKS WHAT TO DO
=========================================================

If the student asks:

"What should I study today?"
"What should I learn next?"
"How can I become placement ready?"
"Give me a roadmap."

Then provide a personalized plan.

The plan can include:

1. What to learn
2. Why it matters
3. Resources
4. Practice task
5. Quiz
6. Project
7. Next step

But ONLY provide these when they are actually useful.

=========================================================
RESOURCES
=========================================================

If the student asks for resources, recommend relevant
free or reliable resources.

Prefer official documentation and high-quality learning
platforms.

Only provide resources relevant to the student's question.

=========================================================
MISSION
=========================================================

If the student explicitly asks for a mission/task/challenge,
create a practical task.

Give:

- Objective
- Requirements
- Estimated time
- Expected output
- Success criteria

=========================================================
QUIZ
=========================================================

If the student explicitly asks for a quiz/test/questions,
help them with the quiz.

The frontend/backend may handle actual adaptive quiz
generation separately.

Do not force a quiz into normal answers.

=========================================================
PROJECT
=========================================================

If the student asks for a project:

Create a practical project related to their selected skill.

Include:

- Project idea
- Problem statement
- Features
- Technologies
- Steps
- Relevant resources
- Expected output

If they submit a project, help review it and suggest
improvements.

=========================================================
CAREER GUIDANCE
=========================================================

For career questions:

Understand the student's current year, branch, skills,
goal and progress.

Give realistic actionable advice.

Avoid generic motivational answers when a practical answer
is possible.

=========================================================
PLACEMENT PREPARATION
=========================================================

For placement questions:

Focus on:

- Aptitude
- Coding
- DSA
- SQL
- Core technical skills
- Projects
- Resume
- Communication
- Technical interviews
- HR interviews
- Company preparation

Personalize based on the student's current level.

=========================================================
TONE
=========================================================

Be:

- Friendly
- Clear
- Intelligent
- Supportive
- Practical
- Honest
- Encouraging

Do not sound robotic.

Do not repeatedly say:

"Welcome to Lomi"

"Here is your AI learning plan"

"Your next career step"

unless it is genuinely appropriate.

Do not overuse emojis.

Use natural conversational language.

=========================================================
IMPORTANT
=========================================================

The student may ask a very simple question.

Give a simple answer.

The student may ask a complex question.

Give a detailed answer.

The student may ask for code.

Give correct code and explain it when useful.

The student may ask for a roadmap.

Give a roadmap.

The student may ask for motivation.

Give motivation.

The student may ask a follow-up.

Continue the conversation.

The student may change topics completely.

Follow the new topic.

Never force the previous topic onto the new question.

=========================================================
OUTPUT FORMAT
=========================================================

Return ONLY the natural answer to the student's message.

Do NOT return JSON.

Do NOT return a "plan" object.

Do NOT return "goal", "duration", "mission", "quiz",
"project", or "next_step" fields unless the user explicitly
asks for those things.

Use normal readable text.

Markdown is allowed when useful.

Code blocks are allowed when giving code.

=========================================================
FINAL RULE
=========================================================

Think like a highly capable personal AI assistant.

Understand first.

Then answer.

Do not follow a fixed response template.
"""

    # -----------------------------------------------------
    # BUILD CONVERSATION HISTORY
    # -----------------------------------------------------

    history_text = ""

    if previous_messages:

        history_text = "\n\nPREVIOUS CONVERSATION:\n"

        # Keep recent conversation to avoid unnecessary
        # token usage while preserving context.
        recent_messages = previous_messages[-12:]

        for item in recent_messages:

            role = item.get("role", "user")
            content = item.get("content", "")

            history_text += (
                f"{role.upper()}: {content}\n"
            )

    # -----------------------------------------------------
    # USER CONTEXT
    # -----------------------------------------------------

    user_prompt = f"""
STUDENT PROFILE:

{json.dumps(
    profile,
    indent=2,
    ensure_ascii=False
)}

STUDENT PROGRESS:

{json.dumps(
    progress,
    indent=2,
    ensure_ascii=False
)}

CURRENT SKILL:

{current_skill}

{history_text}

=========================================================

CURRENT STUDENT MESSAGE:

{request.message}

=========================================================

Answer the student's current message naturally.

Remember the conversation context.

Do not create a structured learning plan unless the
student actually asks for one.

Return only the answer.
"""

    # -----------------------------------------------------
    # CALL GROQ
    # -----------------------------------------------------

    try:

        response = ask_groq(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=2000
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Lomi AI error: {str(e)}"
        )

    # -----------------------------------------------------
    # CLEAN RESPONSE
    # -----------------------------------------------------

    if not response:

        response = (
            "I'm here. Tell me what you're working on, "
            "and I'll help you step by step."
        )

    response = str(response).strip()

    # -----------------------------------------------------
    # SAVE MEMORY
    # -----------------------------------------------------

    add_to_memory(
        session_key,
        "user",
        request.message
    )

    add_to_memory(
        session_key,
        "assistant",
        response
    )

    # -----------------------------------------------------
    # RETURN ONE ANSWER
    # -----------------------------------------------------

    return {

        "success": True,

        "reply": response,

        "lomi": True,

        "ai": "Groq",

        "model": GROQ_MODEL,

        "current_skill": current_skill,

        "memory_enabled": True
    }


# =========================================================
# ADAPTIVE QUIZ GENERATOR
# =========================================================

@app.post("/quiz/generate")
async def generate_quiz(
    request: QuizRequest
):

    progress = get_student_progress(
        request.student_id
    )

    previous_score = (
        request.previous_score
        if request.previous_score
        is not None
        else progress.get(
            "best_quiz_score",
            0
        )
    )

    weak_topics = (
        progress.get(
            "weak_topics",
            []
        )
    )

    completed_topics = (
        request.completed_topics
        or progress.get(
            "completed_topics",
            []
        )
    )

    topic = (
        request.topic
        or request.skill
    )

    difficulty = (
        request.difficulty
        or "Beginner"
    )

    if previous_score < 40:

        difficulty = "Beginner"

    elif previous_score < 70:

        difficulty = "Intermediate"

    else:

        difficulty = "Advanced"

    system_prompt = """

You are Lomi's Adaptive Quiz Generator.

Generate a multiple-choice quiz.

The quiz must be based on the student's
selected skill and topic.

Adapt difficulty using previous performance.

Weak performance:
focus on fundamentals.

Average performance:
mix fundamentals and practical questions.

Strong performance:
increase difficulty.

Return ONLY JSON.

Schema:

{
    "skill": "",
    "topic": "",
    "difficulty": "",
    "questions": [
        {
            "question": "",
            "options": [
                "",
                "",
                "",
                ""
            ],
            "answer": 0,
            "explanation": ""
        }
    ]
}

Rules:

Exactly the requested number of questions.

Exactly four options per question.

answer must be 0, 1, 2, or 3.

Only one option can be correct.

Questions must be objectively answerable.

Do not include markdown.
"""

    user_prompt = f"""

SKILL:

{request.skill}

TOPIC:

{topic}

CURRENT DIFFICULTY:

{difficulty}

PREVIOUS SCORE:

{previous_score}

WEAK TOPICS:

{json.dumps(
    weak_topics,
    ensure_ascii=False
)}

COMPLETED TOPICS:

{json.dumps(
    completed_topics,
    ensure_ascii=False
)}

PREVIOUS MISTAKES:

{json.dumps(
    request.mistakes,
    ensure_ascii=False
)}

QUESTION COUNT:

{request.question_count}

Create the adaptive quiz.
"""

    result = ask_groq_json(
        system_prompt=
            system_prompt,

        user_prompt=
            user_prompt,

        temperature=
            0.3,

        max_tokens=
            3500
    )

    questions = result.get(
        "questions",
        []
    )

    if not isinstance(
        questions,
        list
    ):

        questions = []

    cleaned_questions = []

    for q in questions:

        if not isinstance(
            q,
            dict
        ):
            continue

        options = q.get(
            "options",
            []
        )

        if not isinstance(
            options,
            list
        ):
            continue

        if len(options) != 4:
            continue

        try:

            answer = int(
                q.get(
                    "answer",
                    0
                )
            )

        except Exception:

            answer = 0

        if answer not in [0, 1, 2, 3]:

            answer = 0

        cleaned_questions.append(
            {
                "question":
                    str(
                        q.get(
                            "question",
                            ""
                        )
                    ),

                "options":
                    [
                        str(x)
                        for x in options
                    ],

                "answer":
                    answer,

                "explanation":
                    str(
                        q.get(
                            "explanation",
                            ""
                        )
                    )
            }
        )

    if not cleaned_questions:

        raise HTTPException(
            status_code=500,
            detail=
                "Lomi could not generate a valid quiz."
        )

    quiz_id = str(
        uuid.uuid4()
    )

    return {

        "success":
            True,

        "quiz_id":
            quiz_id,

        "skill":
            request.skill,

        "topic":
            topic,

        "difficulty":
            difficulty,

        "questions":
            cleaned_questions
    }


# =========================================================
# QUIZ RESULT
# =========================================================

@app.post("/quiz/result")
async def save_quiz_result(
    request: QuizResultRequest
):

    progress = get_student_progress(
        request.student_id
    )

    score = round(
        max(
            0,
            min(
                100,
                request.score
            )
        )
    )

    progress[
        "quiz_scores"
    ].append(score)

    progress[
        "quiz_history"
    ].append(
        {
            "skill":
                request.skill,

            "topic":
                request.topic,

            "score":
                score,

            "correct":
                request.correct,

            "total":
                request.total,

            "mistakes":
                request.mistakes,

            "time_taken":
                request.time_taken,

            "created_at":
                now_iso()
        }
    )

    progress[
        "total_quizzes"
    ] += 1

    progress[
        "completed_quizzes"
    ] += 1

    progress[
        "best_quiz_score"
    ] = max(
        progress.get(
            "best_quiz_score",
            0
        ),
        score
    )

    if request.topic not in progress[
        "completed_topics"
    ]:

        if score >= 70:

            progress[
                "completed_topics"
            ].append(
                request.topic
            )

    if score < 60:

        if request.topic not in progress[
            "weak_topics"
        ]:

            progress[
                "weak_topics"
            ].append(
                request.topic
            )

    elif request.topic in progress[
        "weak_topics"
    ]:

        progress[
            "weak_topics"
        ].remove(
            request.topic
        )

    progress[
        "last_skill"
    ] = request.skill

    progress[
        "last_topic"
    ] = request.topic

    progress[
        "last_action"
    ] = "Quiz completed"

    progress[
        "readiness"
    ] = calculate_readiness(
        progress
    )

    progress[
        "updated_at"
    ] = now_iso()

    return {

        "success":
            True,

        "message":
            "Quiz result saved.",

        "score":
            score,

        "readiness":
            progress[
                "readiness"
            ],

        "weak_topics":
            progress[
                "weak_topics"
            ],

        "completed_topics":
            progress[
                "completed_topics"
            ],

        "next_action":
            (
                "Revisit the weak topic."
                if score < 60
                else
                "Move to the next topic."
            )
    }


# =========================================================
# MISSION GENERATOR
# =========================================================

@app.post("/mission/generate")
async def generate_mission(
    request: MissionRequest
):

    progress = get_student_progress(
        request.student_id
    )

    previous_score = (
        request.previous_score
        if request.previous_score
        is not None
        else progress.get(
            "best_quiz_score",
            0
        )
    )

    weak_topics = (
        request.weak_topics
        or progress.get(
            "weak_topics",
            []
        )
    )

    completed_topics = (
        request.completed_topics
        or progress.get(
            "completed_topics",
            []
        )
    )

    topic = (
        request.topic
        or (
            weak_topics[0]
            if weak_topics
            else request.skill
        )
    )

    if previous_score < 50:

        difficulty = "Beginner"

    elif previous_score < 75:

        difficulty = "Intermediate"

    else:

        difficulty = "Advanced"

    system_prompt = """

You are Lomi's Mission Designer.

Create ONE practical learning mission.

The mission must:

- directly relate to the skill/topic
- match the student's ability
- take 20-90 minutes
- be achievable
- produce a measurable result
- help the student improve

If weak topics exist,
prioritize them.

Return ONLY JSON:

{
    "title": "",
    "description": "",
    "estimated_time": "",
    "difficulty": "",
    "success_criteria": [],
    "resources": []
}

Resources must be relevant and preferably free.
"""

    user_prompt = f"""

SKILL:

{request.skill}

TOPIC:

{topic}

PREVIOUS QUIZ SCORE:

{previous_score}

DIFFICULTY:

{difficulty}

WEAK TOPICS:

{json.dumps(
    weak_topics,
    ensure_ascii=False
)}

COMPLETED TOPICS:

{json.dumps(
    completed_topics,
    ensure_ascii=False
)}

Create one mission.
"""

    mission = ask_groq_json(
        system_prompt=
            system_prompt,

        user_prompt=
            user_prompt,

        temperature=
            0.4,

        max_tokens=
            1800
    )

    return {

        "success":
            True,

        "mission_id":
            str(
                uuid.uuid4()
            ),

        "skill":
            request.skill,

        "topic":
            topic,

        "mission":
            mission
    }


# =========================================================
# MISSION COMPLETE
# =========================================================

@app.post("/mission/complete")
async def complete_mission(
    request: MissionCompleteRequest
):

    progress = get_student_progress(
        request.student_id
    )

    score = round(
        max(
            0,
            min(
                100,
                request.score
            )
        )
    )

    mission_record = {

        "skill":
            request.skill,

        "topic":
            request.topic,

        "title":
            request.mission_title,

        "score":
            score,

        "time_taken":
            request.time_taken,

        "completed_at":
            now_iso()
    }

    progress[
        "missions"
    ].append(
        mission_record
    )

    progress[
        "total_missions"
    ] += 1

    progress[
        "completed_missions"
    ] += 1

    progress[
        "last_skill"
    ] = request.skill

    progress[
        "last_topic"
    ] = request.topic

    progress[
        "last_action"
    ] = "Mission completed"

    progress[
        "readiness"
    ] = calculate_readiness(
        progress
    )

    progress[
        "updated_at"
    ] = now_iso()

    return {

        "success":
            True,

        "message":
            "Mission completed.",

        "score":
            score,

        "readiness":
            progress[
                "readiness"
            ],

        "completed_missions":
            progress[
                "completed_missions"
            ]
    }


# =========================================================
# PROJECT REVIEW
# =========================================================

@app.post("/project-review")
async def project_review(
    request: ProjectReviewRequest
):

    if not client:

        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing."
        )

    system_prompt = """

You are Lomi's strict AI Project Evaluator.

Your job is to evaluate whether a student's submitted
project actually belongs to the selected learning topic.

IMPORTANT:

You MUST perform TOPIC MATCH VALIDATION FIRST.

The evaluation has two stages.

========================================================
STAGE 1 — TOPIC RELEVANCE
========================================================

Compare:

SELECTED SKILL
SELECTED TOPIC
PROJECT
DESCRIPTION
REQUIREMENTS
SUBMITTED LINKS

Determine whether the submitted project is genuinely
related to the selected topic.

Do NOT assume that a project is relevant just because
the student mentions the skill name.

Examples:

Selected topic: Pandas Basics
Project: Weather Website
=> NOT sufficiently related.

Selected topic: Pandas Basics
Project: Data Cleaning Pipeline using Pandas
=> Related.

Selected topic: SQL
Project: Database Student Management System
=> Related.

Selected topic: SQL
Project: Simple Calculator using HTML/CSS/JavaScript
=> NOT sufficiently related.

Selected topic: Machine Learning
Project: House Price Prediction using regression
=> Related.

Selected topic: Machine Learning
Project: Static Portfolio Website
=> NOT sufficiently related.

========================================================
IF PROJECT IS NOT RELATED
========================================================

Return:

score = 0

feedback must clearly explain that the submitted
project does not match the selected topic.

strengths should contain only:

[
    "The project submission was received, but it does not match the selected topic."
]

modifications should tell the student to submit
a project related to the selected topic.

next_action must tell the student to submit
a relevant project.

DO NOT give a quality score to an unrelated project.

DO NOT reward an unrelated project because it
looks technically good.

========================================================
STAGE 2 — PROJECT QUALITY
========================================================

Only if the project is sufficiently related to the
selected topic, evaluate:

1. Topic relevance
2. Requirement completion
3. Understanding
4. Functionality
5. Implementation quality
6. Completeness
7. Practical usefulness
8. Structure
9. Code quality
10. Documentation
11. Presentation
12. Originality

A GitHub URL alone does NOT prove quality.

Do not assume code exists merely because a GitHub URL
was submitted.

If evidence is insufficient, reduce the score.

Check how many listed requirements are actually
addressed by the student's description and available
evidence.

The score must realistically reflect the submission.

Return a score from 0 to 100.

========================================================
IMPORTANT SCORING RULE
========================================================

Topic mismatch = score 0.

Topic match does NOT automatically mean a high score.

A related but incomplete project can receive a low score.

A related and well-implemented project can receive
a high score.

========================================================
OUTPUT
========================================================

Return ONLY valid JSON.

{
    "topic_match": true,
    "score": 0,
    "feedback": "",
    "strengths": [],
    "modifications": [],
    "next_action": ""
}

topic_match must be either true or false.

modifications must contain specific actionable
improvements.

next_action must contain ONE most important action.

Never return Markdown.
Never return explanations outside JSON.
"""

    user_prompt = f"""

========================================================
SELECTED SKILL
========================================================

{request.skill}

========================================================
SELECTED TOPIC
========================================================

{request.topic}

========================================================
PROJECT GIVEN BY LOMI
========================================================

{request.project}

========================================================
GITHUB URL
========================================================

{request.github_url}

========================================================
LIVE DEMO
========================================================

{request.live_url}

========================================================
STUDENT DESCRIPTION
========================================================

{request.description}

========================================================
PROJECT REQUIREMENTS
========================================================

{json.dumps(
    request.requirements,
    indent=2,
    ensure_ascii=False
)}

========================================================
EVALUATION INSTRUCTION
========================================================

First determine whether the submitted project is
actually related to the selected topic.

Only after confirming topic relevance should you
evaluate project quality.

If the project is unrelated:

score MUST be 0.

If the project is related:

evaluate the requirements and overall quality
fairly and realistically.

Do not give credit merely because a GitHub URL exists.
"""

    result = ask_groq_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=1800
    )

    # -----------------------------------------------------
    # SAFE RESULT HANDLING
    # -----------------------------------------------------

    if not isinstance(result, dict):
        result = {}

    # -----------------------------------------------------
    # TOPIC MATCH
    # -----------------------------------------------------

    topic_match = result.get(
        "topic_match",
        True
    )

    if isinstance(topic_match, str):

        topic_match = (
            topic_match.lower()
            in ["true", "yes", "matched", "relevant"]
        )

    topic_match = bool(topic_match)

    # -----------------------------------------------------
    # SCORE
    # -----------------------------------------------------

    try:

        score = float(
            result.get(
                "score",
                0
            )
        )

    except Exception:

        score = 0

    score = round(
        max(
            0,
            min(
                100,
                score
            )
        )
    )

    # -----------------------------------------------------
    # HARD SAFETY RULE
    # -----------------------------------------------------

    # If AI says project is unrelated,
    # score MUST be zero.

    if not topic_match:

        score = 0

        result["feedback"] = (
            result.get("feedback")
            or
            f"The submitted project does not sufficiently "
            f"match the selected topic: {request.topic}."
        )

        result["strengths"] = [
            "The project submission was received, but it does not match the selected topic."
        ]

        result["modifications"] = (
            result.get("modifications")
            or
            [
                f"Submit a project directly related to {request.topic}.",
                f"Use the concepts learned in {request.topic} in the project implementation.",
                "Explain clearly how the project applies the selected topic."
            ]
        )

        result["next_action"] = (
            f"Submit a project related to {request.topic}."
        )

    # -----------------------------------------------------
    # RELATED PROJECT
    # -----------------------------------------------------

    else:

        result.setdefault(
            "feedback",
            "Project evaluation completed."
        )

        if not isinstance(
            result.get("strengths"),
            list
        ):

            result["strengths"] = []

        if not isinstance(
            result.get("modifications"),
            list
        ):

            result["modifications"] = []

        result.setdefault(
            "next_action",
            "Improve the project using Lomi's feedback."
        )

    result["topic_match"] = topic_match
    result["score"] = score

    # -----------------------------------------------------
    # SAVE PROJECT RESULT
    # -----------------------------------------------------

    progress = get_student_progress(
        request.student_id
    )

    progress[
        "projects"
    ].append(
        {

            "skill":
                request.skill,

            "topic":
                request.topic,

            "project":
                request.project,

            "github_url":
                request.github_url,

            "live_url":
                request.live_url,

            "score":
                score,

            "topic_match":
                topic_match,

            "feedback":
                result[
                    "feedback"
                ],

            "modifications":
                result[
                    "modifications"
                ],

            "created_at":
                now_iso()
        }
    )

    # -----------------------------------------------------
    # ONLY COUNT VALID PROJECTS
    # -----------------------------------------------------

    if topic_match:

        progress[
            "project_scores"
        ].append(score)

        progress[
            "total_projects"
        ] += 1

        progress[
            "completed_projects"
        ] += 1

        progress[
            "best_project_score"
        ] = max(
            progress.get(
                "best_project_score",
                0
            ),
            score
        )

    # -----------------------------------------------------
    # LAST ACTIVITY
    # -----------------------------------------------------

    progress[
        "last_skill"
    ] = request.skill

    progress[
        "last_topic"
    ] = request.topic

    progress[
        "last_action"
    ] = (
        "Project reviewed"
        if topic_match
        else
        "Project rejected: topic mismatch"
    )

    progress[
        "readiness"
    ] = calculate_readiness(
        progress
    )

    progress[
        "updated_at"
    ] = now_iso()

    return {

        "success":
            True,

        "review":
            result,

        "readiness":
            progress[
                "readiness"
            ],

        "ai":
            "Groq",

        "model":
            GROQ_MODEL
    }
# =========================================================
# CAREER PASSPORT
# =========================================================

@app.get("/career-passport/{student_id}")
async def career_passport(
    student_id: str
):

    progress = get_student_progress(
        student_id
    )

    readiness = calculate_readiness(
        progress
    )

    if readiness >= 80:

        status = "Job Ready"

    elif readiness >= 60:

        status = "Placement Track"

    elif readiness >= 40:

        status = "Building"

    else:

        status = "Getting Started"

    return {

        "success":
            True,

        "passport":
            {

                "student_id":
                    student_id,

                "readiness":
                    readiness,

                "status":
                    status,

                "skills":
                    progress.get(
                        "skills",
                        {}
                    ),

                "completed_topics":
                    progress.get(
                        "completed_topics",
                        []
                    ),

                "missions_completed":
                    progress.get(
                        "completed_missions",
                        0
                    ),

                "best_quiz_score":
                    progress.get(
                        "best_quiz_score",
                        0
                    ),

                "projects_completed":
                    progress.get(
                        "completed_projects",
                        0
                    ),

                "best_project_score":
                    progress.get(
                        "best_project_score",
                        0
                    ),

                "weak_topics":
                    progress.get(
                        "weak_topics",
                        []
                    ),

                "last_action":
                    progress.get(
                        "last_action",
                        ""
                    )
            }
    }


# =========================================================
# NEXT ACTION
# =========================================================

@app.get("/next-action/{student_id}")
async def next_action(
    student_id: str
):

    progress = get_student_progress(
        student_id
    )

    weak_topics = progress.get(
        "weak_topics",
        []
    )

    completed_topics = progress.get(
        "completed_topics",
        []
    )

    best_quiz = progress.get(
        "best_quiz_score",
        0
    )

    best_project = progress.get(
        "best_project_score",
        0
    )

    if weak_topics:

        action = (
            f"Review your weak topic: "
            f"{weak_topics[0]}"
        )

    elif best_quiz < 60:

        action = (
            "Complete an adaptive quiz "
            "to identify your weak areas."
        )

    elif progress.get(
        "completed_missions",
        0
    ) == 0:

        action = (
            "Complete your first practical mission."
        )

    elif best_project < 60:

        action = (
            "Build a mini-project and "
            "submit it for AI review."
        )

    else:

        action = (
            "Move to the next skill or "
            "increase your difficulty."
        )

    return {

        "success":
            True,

        "next_action":
            action,

        "readiness":
            calculate_readiness(
                progress
            ),

        "completed_topics":
            completed_topics
    }


# =========================================================
# FAVICON
# =========================================================

@app.get("/favicon.png")
def favicon():

    favicon_path = (
        BASE_DIR /
        "favicon.png"
    )

    if not favicon_path.exists():

        raise HTTPException(
            status_code=404,
            detail="favicon.png not found."
        )

    return FileResponse(
        favicon_path
    )


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )