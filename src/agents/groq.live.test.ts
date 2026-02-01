import { completeSimple, type Model } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import { isTruthyEnvValue } from "../infra/env.js";

const GROQ_KEY = process.env.GROQ_API_KEY ?? "";
const GROQ_BASE_URL = process.env.GROQ_BASE_URL?.trim() || "https://api.groq.com/openai/v1";
const GROQ_MODEL = process.env.GROQ_MODEL?.trim() || "llama-3.1-8b-instant";
const LIVE = isTruthyEnvValue(process.env.GROQ_LIVE_TEST) || isTruthyEnvValue(process.env.LIVE);

const describeLive = LIVE && GROQ_KEY ? describe : describe.skip;

describeLive("groq live", () => {
  it("returns assistant text", async () => {
    const model: Model<"openai-completions"> = {
      id: GROQ_MODEL,
      name: `Groq ${GROQ_MODEL}`,
      api: "openai-completions",
      provider: "groq",
      baseUrl: GROQ_BASE_URL,
      reasoning: false,
      input: ["text"],
      cost: { input: 0.05, output: 0.08, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 131072,
      maxTokens: 8192,
    };
    const res = await completeSimple(
      model,
      {
        messages: [
          {
            role: "user",
            content: "Reply with the word ok.",
            timestamp: Date.now(),
          },
        ],
      },
      { apiKey: GROQ_KEY, maxTokens: 64 },
    );
    const text = res.content
      .filter((block) => block.type === "text")
      .map((block) => block.text.trim())
      .join(" ");
    expect(text.length).toBeGreaterThan(0);
    expect(text.toLowerCase()).toContain("ok");
  }, 20000);

  it("handles multi-turn conversation", async () => {
    const model: Model<"openai-completions"> = {
      id: GROQ_MODEL,
      name: `Groq ${GROQ_MODEL}`,
      api: "openai-completions",
      provider: "groq",
      baseUrl: GROQ_BASE_URL,
      reasoning: false,
      input: ["text"],
      cost: { input: 0.05, output: 0.08, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 131072,
      maxTokens: 8192,
    };
    const res = await completeSimple(
      model,
      {
        messages: [
          {
            role: "user",
            content: "My name is Alice.",
            timestamp: Date.now(),
          },
          {
            role: "assistant",
            content: [{ type: "text", text: "Hello Alice!" }],
            timestamp: Date.now(),
          },
          {
            role: "user",
            content: "What is my name?",
            timestamp: Date.now(),
          },
        ],
      },
      { apiKey: GROQ_KEY, maxTokens: 64 },
    );
    const text = res.content
      .filter((block) => block.type === "text")
      .map((block) => block.text.trim())
      .join(" ");
    expect(text.toLowerCase()).toContain("alice");
  }, 20000);
});
