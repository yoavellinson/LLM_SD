import asyncio
import time
import json
from openai import AsyncOpenAI
from openai_key import key
client = AsyncOpenAI(api_key=key)

PROMPT = "Return ONLY this JSON list: [0,1,0,1,0]"

async def one_call(i):
    r = await client.responses.create(
        model="gpt-4o-mini",
        input=PROMPT,
        temperature=0,
    )
    return r.output_text

async def main():
    t0 = time.time()
    tasks = [one_call(i) for i in range(10)]  # 10 calls in parallel
    outs = await asyncio.gather(*tasks)
    print("Outputs:", outs)
    print("Total time:", time.time() - t0)

asyncio.run(main())
