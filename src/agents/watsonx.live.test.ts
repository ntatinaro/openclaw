import type { Model } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import { isTruthyEnvValue } from "../infra/env.js";
import { streamWatsonx } from "../providers/watsonx-provider.js";

const WATSONX_KEY = process.env.WATSONX_API_KEY ?? "";
const WATSONX_PROJECT_ID = process.env.WATSONX_PROJECT_ID ?? "";
const WATSONX_BASE_URL =
  process.env.WATSONX_BASE_URL?.trim() || "https://us-south.ml.cloud.ibm.com";
const WATSONX_MODEL = process.env.WATSONX_MODEL?.trim() || "ibm/granite-3-8b-instruct";
const LIVE = isTruthyEnvValue(process.env.WATSONX_LIVE_TEST) || isTruthyEnvValue(process.env.LIVE);

const describeLive = LIVE && WATSONX_KEY && WATSONX_PROJECT_ID ? describe : describe.skip;

describeLive("watsonx live", () => {
  it("streams text generation", async () => {
    const model: Model<"watsonx-generation"> = {
      id: WATSONX_MODEL,
      name: `Watsonx ${WATSONX_MODEL}`,
      api: "watsonx-generation",
      provider: "watsonx",
      baseUrl: WATSONX_BASE_URL,
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 8192,
      maxTokens: 4096,
    };

    const stream = streamWatsonx(
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
      {
        apiKey: WATSONX_KEY,
        projectId: WATSONX_PROJECT_ID,
        maxTokens: 64,
      },
    );

    let fullText = "";
    let gotDone = false;

    for await (const event of stream) {
      if (event.type === "text_delta") {
        fullText += event.delta;
      }
      if (event.type === "text_end") {
        fullText = event.content;
      }
      if (event.type === "done") {
        gotDone = true;
        expect(event.message.usage).toBeDefined();
        expect(event.message.usage.totalTokens).toBeGreaterThan(0);
      }
    }

    expect(fullText.length).toBeGreaterThan(0);
    expect(gotDone).toBe(true);
  }, 30000);

  it("handles system prompt", async () => {
    const model: Model<"watsonx-generation"> = {
      id: WATSONX_MODEL,
      name: `Watsonx ${WATSONX_MODEL}`,
      api: "watsonx-generation",
      provider: "watsonx",
      baseUrl: WATSONX_BASE_URL,
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 8192,
      maxTokens: 4096,
    };

    const stream = streamWatsonx(
      model,
      {
        systemPrompt: "You are a helpful assistant. Always respond in lowercase.",
        messages: [
          {
            role: "user",
            content: "Say hello.",
            timestamp: Date.now(),
          },
        ],
      },
      {
        apiKey: WATSONX_KEY,
        projectId: WATSONX_PROJECT_ID,
        maxTokens: 64,
      },
    );

    let fullText = "";
    for await (const event of stream) {
      if (event.type === "text_end") {
        fullText = event.content;
      }
    }

    expect(fullText.length).toBeGreaterThan(0);
  }, 30000);

  it("handles multi-turn conversation", async () => {
    const model: Model<"watsonx-generation"> = {
      id: WATSONX_MODEL,
      name: `Watsonx ${WATSONX_MODEL}`,
      api: "watsonx-generation",
      provider: "watsonx",
      baseUrl: WATSONX_BASE_URL,
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 8192,
      maxTokens: 4096,
    };

    const stream = streamWatsonx(
      model,
      {
        messages: [
          {
            role: "user",
            content: "My favorite color is blue.",
            timestamp: Date.now(),
          },
          {
            role: "assistant",
            content: [{ type: "text", text: "Got it, you like blue!" }],
            api: "watsonx-generation",
            provider: "watsonx",
            model: WATSONX_MODEL,
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: "stop",
            timestamp: Date.now(),
          },
          {
            role: "user",
            content: "What is my favorite color?",
            timestamp: Date.now(),
          },
        ],
      },
      {
        apiKey: WATSONX_KEY,
        projectId: WATSONX_PROJECT_ID,
        maxTokens: 64,
      },
    );

    let fullText = "";
    for await (const event of stream) {
      if (event.type === "text_end") {
        fullText = event.content;
      }
    }

    expect(fullText.toLowerCase()).toContain("blue");
  }, 30000);
});
