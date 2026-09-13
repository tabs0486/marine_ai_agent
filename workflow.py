from groq import Groq


def run_rag_workflow(
    client,
    manufacturer,
    engine_model,
    serial_number,
    defect,
    question,
    context
):

    """
    Multi-stage RAG workflow.

    Stage 1:
    Prepare engine information.

    Stage 2:
    Provide retrieved manual evidence to the LLM.

    Stage 3:
    Generate structured troubleshooting guidance.

    Stage 4:
    Return the final answer.
    """

    prompt = build_workflow_prompt(
        manufacturer=manufacturer,
        engine_model=engine_model,
        serial_number=serial_number,
        defect=defect,
        question=question,
        context=context
    )

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {
                "role": "system",
                "content": """
You are a safety-conscious professional marine
engine troubleshooting assistant.

You must ground your response in the supplied
technical-manual evidence.

Never fabricate technical specifications.
"""
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1,

        max_tokens=3000
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    if not answer:

        raise RuntimeError(
            "The Groq model returned an empty response."
        )

    return answer


def build_workflow_prompt(
    manufacturer,
    engine_model,
    serial_number,
    defect,
    question,
    context
):

    return f"""
You are troubleshooting a marine engine.

ENGINE INFORMATION
-------------------

Manufacturer:
{manufacturer}

Engine Model:
{engine_model}

Serial Number:
{serial_number if serial_number else "Not provided"}


DEFECT
------

{defect if defect else "Not provided"}


USER QUESTION
-------------

{question if question else "Provide a troubleshooting assessment."}


RETRIEVED MANUAL INFORMATION
----------------------------

{context}


TASK
----

Analyze the problem using the supplied manual evidence.

Produce the following:

1. Problem assessment

2. Most likely causes

3. Troubleshooting sequence

4. Relevant manual evidence

5. Missing information

6. Safety / escalation guidance


IMPORTANT:

- Do not invent technical specifications.
- Do not invent alarm meanings.
- Do not invent pressure/temperature/torque limits.
- Do not claim certainty without evidence.
- Clearly distinguish manual evidence from engineering interpretation.
- If information is absent from the supplied manual evidence,
  explicitly state that it is unavailable.
- Never recommend bypassing alarms, trips, interlocks or
  safety systems.
- Recommend following OEM and vessel safety procedures.
- Do not recommend operation outside approved limits.
"""
