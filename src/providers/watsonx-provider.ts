/**
 * IBM Watsonx.ai provider implementation
 *
 * Supports both IBM Cloud and on-premises deployments.
 * Uses IBM's IAM token exchange for authentication.
 *
 * API Reference: https://cloud.ibm.com/apidocs/watsonx-ai
 */

import type {
  AssistantMessage,
  Context,
  Model,
  StreamFunction,
  StreamOptions,
  TextContent,
} from "@mariozechner/pi-ai/dist/types.js";
import { registerApiProvider, type ApiProvider } from "@mariozechner/pi-ai/dist/api-registry.js";
import { AssistantMessageEventStream } from "@mariozechner/pi-ai/dist/utils/event-stream.js";

const IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token";
const API_VERSION = "2024-03-14";
const DEFAULT_MAX_TOKENS = 4096;
const DEFAULT_TEMPERATURE = 0.7;

// Token cache to avoid repeated IAM token exchanges
interface TokenCache {
  token: string;
  expiresAt: number;
}
const tokenCache = new Map<string, TokenCache>();

export interface WatsonxStreamOptions extends StreamOptions {
  /** IBM Cloud project ID (required for cloud deployments) */
  projectId?: string;
  /** Custom parameters for the model */
  parameters?: {
    decoding_method?: "greedy" | "sample";
    top_p?: number;
    top_k?: number;
    repetition_penalty?: number;
    stop_sequences?: string[];
  };
}

/**
 * Exchange IBM Cloud API key for IAM bearer token
 */
async function getIamToken(apiKey: string, signal?: AbortSignal): Promise<string> {
  const cacheKey = apiKey.slice(0, 16); // Use prefix as cache key
  const cached = tokenCache.get(cacheKey);

  // Return cached token if still valid (with 5 minute buffer)
  if (cached && cached.expiresAt > Date.now() + 300_000) {
    return cached.token;
  }

  const response = await fetch(IAM_TOKEN_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Accept: "application/json",
    },
    body: new URLSearchParams({
      grant_type: "urn:ibm:params:oauth:grant-type:apikey",
      apikey: apiKey,
    }),
    signal,
  });

  if (!response.ok) {
    const error = await response.text().catch(() => "Unknown error");
    throw new Error(`IBM IAM token exchange failed: ${response.status} - ${error}`);
  }

  const data = (await response.json()) as { access_token: string; expires_in: number };
  const expiresAt = Date.now() + data.expires_in * 1000;

  tokenCache.set(cacheKey, { token: data.access_token, expiresAt });
  return data.access_token;
}

/**
 * Convert OpenClaw context to Watsonx prompt format
 */
function buildWatsonxPrompt(context: Context): string {
  const parts: string[] = [];

  // Add system message if present
  if (context.systemPrompt?.trim()) {
    parts.push(`<|system|>\n${context.systemPrompt}\n<|end|>`);
  }

  // Convert messages to prompt format
  for (const msg of context.messages) {
    if (msg.role === "user") {
      const content =
        typeof msg.content === "string"
          ? msg.content
          : msg.content
              .filter((c): c is TextContent => c.type === "text")
              .map((c) => c.text)
              .join("\n");
      parts.push(`<|user|>\n${content}\n<|end|>`);
    } else if (msg.role === "assistant") {
      const content = msg.content
        .filter((c): c is TextContent => c.type === "text")
        .map((c) => c.text)
        .join("\n");
      parts.push(`<|assistant|>\n${content}\n<|end|>`);
    }
  }

  // Add assistant prefix for response
  parts.push("<|assistant|>\n");

  return parts.join("\n");
}

/**
 * Stream text generation from Watsonx
 */
export const streamWatsonx: StreamFunction<"watsonx-generation", WatsonxStreamOptions> = (
  model: Model<"watsonx-generation">,
  context: Context,
  options?: WatsonxStreamOptions,
) => {
  const stream = new AssistantMessageEventStream();

  // Build the output message that we'll update as we stream
  const output: AssistantMessage = {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
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
  };

  (async () => {
    try {
      const apiKey = options?.apiKey ?? process.env.WATSONX_API_KEY;
      if (!apiKey) {
        throw new Error("Watsonx API key required. Set WATSONX_API_KEY or pass apiKey option.");
      }

      const projectId = options?.projectId ?? process.env.WATSONX_PROJECT_ID;
      if (!projectId) {
        throw new Error(
          "Watsonx project ID required. Set WATSONX_PROJECT_ID or pass projectId option.",
        );
      }

      // Get IAM token
      const token = await getIamToken(apiKey, options?.signal);

      // Build request
      const baseUrl = model.baseUrl ?? "https://us-south.ml.cloud.ibm.com";
      const url = `${baseUrl}/ml/v1/text/generation_stream?version=${API_VERSION}`;

      const prompt = buildWatsonxPrompt(context);
      const body = {
        model_id: model.id,
        input: prompt,
        project_id: projectId,
        parameters: {
          max_new_tokens: options?.maxTokens ?? DEFAULT_MAX_TOKENS,
          temperature: options?.temperature ?? DEFAULT_TEMPERATURE,
          ...options?.parameters,
        },
      };

      const response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          ...(options?.headers ?? {}),
        },
        body: JSON.stringify(body),
        signal: options?.signal,
      });

      if (!response.ok) {
        const error = await response.text().catch(() => "Unknown error");
        throw new Error(`Watsonx API error: ${response.status} - ${error}`);
      }

      if (!response.body) {
        throw new Error("Watsonx response body is null");
      }

      // Emit start event
      stream.push({ type: "start", partial: output });

      // Parse SSE stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullText = "";
      let startedText = false;
      const contentIndex = 0;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;

            const data = line.slice(5).trim();
            if (data === "[DONE]") continue;

            try {
              const parsed = JSON.parse(data) as {
                results?: Array<{
                  generated_text?: string;
                  generated_token_count?: number;
                  input_token_count?: number;
                  stop_reason?: string;
                }>;
              };

              const result = parsed.results?.[0];
              if (result?.generated_text) {
                const newText = result.generated_text.slice(fullText.length);
                if (newText) {
                  // Emit text_start on first text
                  if (!startedText) {
                    startedText = true;
                    output.content.push({ type: "text", text: "" });
                    stream.push({
                      type: "text_start",
                      contentIndex,
                      partial: output,
                    });
                  }

                  fullText = result.generated_text;
                  // Update the text content
                  const textBlock = output.content[contentIndex] as TextContent;
                  textBlock.text = fullText;

                  stream.push({
                    type: "text_delta",
                    contentIndex,
                    delta: newText,
                    partial: output,
                  });
                }
              }

              if (result?.input_token_count) {
                output.usage.input = result.input_token_count;
              }
              if (result?.generated_token_count) {
                output.usage.output = result.generated_token_count;
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }
      } finally {
        reader.releaseLock();
      }

      // Update final usage
      output.usage.totalTokens = output.usage.input + output.usage.output;
      const cost = model.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
      output.usage.cost = {
        input: (output.usage.input / 1_000_000) * cost.input,
        output: (output.usage.output / 1_000_000) * cost.output,
        cacheRead: 0,
        cacheWrite: 0,
        total:
          (output.usage.input / 1_000_000) * cost.input +
          (output.usage.output / 1_000_000) * cost.output,
      };

      // Emit text_end if we had text
      if (startedText) {
        stream.push({
          type: "text_end",
          contentIndex,
          content: fullText,
          partial: output,
        });
      }

      // Emit done event
      stream.push({
        type: "done",
        reason: "stop",
        message: output,
      });
    } catch (error) {
      output.stopReason = "error";
      output.errorMessage = error instanceof Error ? error.message : String(error);
      stream.push({
        type: "error",
        reason: "error",
        error: output,
      });
    }
  })();

  return stream;
};

/**
 * Simple streaming (without extended options)
 */
export const streamSimpleWatsonx: StreamFunction<"watsonx-generation", StreamOptions> = (
  model: Model<"watsonx-generation">,
  context: Context,
  options?: StreamOptions,
) => {
  return streamWatsonx(model, context, options as WatsonxStreamOptions);
};

/**
 * Register the Watsonx API provider with pi-ai
 */
export function registerWatsonxProvider(): void {
  const provider: ApiProvider<"watsonx-generation", WatsonxStreamOptions> = {
    api: "watsonx-generation",
    stream: streamWatsonx,
    streamSimple: streamSimpleWatsonx,
  };

  registerApiProvider(provider, "openclaw-watsonx");
}

// Auto-register when module is imported
registerWatsonxProvider();
