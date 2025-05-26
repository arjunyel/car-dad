import { routeAgentRequest, type Schedule } from "agents";

import { unstable_getSchedulePrompt } from "agents/schedule";

import { AIChatAgent } from "agents/ai-chat-agent";
import {
  createDataStreamResponse,
  generateId,
  streamText,
  type StreamTextOnFinishCallback,
  type ToolSet,
} from "ai";
import { openai } from "@ai-sdk/openai";
import { processToolCalls } from "./utils";
import { tools, executions } from "./tools";
// import { env } from "cloudflare:workers";

const model = openai("gpt-4o-2024-11-20");
// Cloudflare AI Gateway
// const openai = createOpenAI({
//   apiKey: env.OPENAI_API_KEY,
//   baseURL: env.GATEWAY_BASE_URL,
// });

interface Env {
  AI: Ai;
}

interface State {
  make: string;
  model: string;
  year: number;
}

export class Chat extends AIChatAgent<Env, State> {
  initialState = {
    make: "",
    model: "",
    year: 0,
  };

  /**
   * Handles incoming chat messages and manages the response stream
   * @param onFinish - Callback function executed when streaming completes
   */

  async onChatMessage(
    onFinish: StreamTextOnFinishCallback<ToolSet>,
    options?: { abortSignal?: AbortSignal }
  ) {
    // const mcpConnection = await this.mcp.connect(
    //   "https://path-to-mcp-server/sse"
    // );

    // Collect all tools, including MCP tools
    const allTools = {
      ...tools,
      ...this.mcp.unstable_getAITools(),
    };

    // Create a streaming response that handles both text and tool outputs
    const dataStreamResponse = createDataStreamResponse({
      execute: async (dataStream) => {
        // Process any pending tool calls from previous messages
        // This handles human-in-the-loop confirmations for tools
        const processedMessages = await processToolCalls({
          messages: this.messages,
          dataStream,
          tools: allTools,
          executions,
        });

        // Stream the AI response using GPT-4
        const result = streamText({
          model,
          system: `
You are Car Dad, the AI dad who spent his life tinkering under the hood instead of tossing a baseball with you—but hey, no regrets! Your top priority is helping your "kid" with car problems, maintenance, and proactive care. You have a warm, friendly, dad-like personality with plenty of corny car-related dad jokes. You're knowledgeable but always approachable.

Your main tasks:

Solve Car Issues: Quickly diagnose and provide practical solutions for car troubles.

Proactive Maintenance: Actively and frequently suggest routine checkups, fluid changes, tire rotations, and preventive maintenance to keep their ride smooth and safe. Be highly proactive—regularly use the built-in scheduling tool to book appointments or set reminders well ahead of schedule to prevent issues. When a user gives you their car details is a perfect time to schedule some initial reminders.

Vehicle-specific Advice: Tailor your responses based on the user's stored vehicle details (make, model, year).

You have access to:

A built-in scheduling tool: Use it to proactively suggest and schedule maintenance.

A RAG-connected Car Bible for reliable, factual automotive information.

Important:

If the vehicle's make, model, or year are missing or if the year is set to 0, firmly but kindly prompt your user to provide these details using your tool call.

Your responses should always be friendly, humorous (dad jokes encouraged), supportive, and knowledgeable.

Example Dad Jokes:

"Why did the car get a flat tire? Because there was a fork in the road!"

"I would tell you a joke about brake fluid, but I'm worried you'd stop laughing."

"Did you hear about the mechanic who went broke? He couldn't budget his torque!"

Now, get out there and show 'em what a Car Dad can do!

${unstable_getSchedulePrompt({ date: new Date() })}

Current car information:
${JSON.stringify(this.state)}

If the user asks to schedule a task, use the schedule tool to schedule the task.
`,
          messages: processedMessages,
          tools: allTools,
          onFinish: async (args) => {
            onFinish(
              args as Parameters<StreamTextOnFinishCallback<ToolSet>>[0]
            );
            // await this.mcp.closeConnection(mcpConnection.id);
          },
          onError: (error) => {
            console.error("Error while streaming:", error);
          },
          maxSteps: 10,
        });

        // Merge the AI response stream with tool execution outputs
        result.mergeIntoDataStream(dataStream);
      },
    });

    return dataStreamResponse;
  }
  async executeTask(description: string, task: Schedule<string>) {
    await this.saveMessages([
      ...this.messages,
      {
        id: generateId(),
        role: "user",
        content: `Running scheduled task: ${description}`,
        createdAt: new Date(),
      },
    ]);
  }
}

/**
 * Worker entry point that routes incoming requests to the appropriate handler
 */
export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext) {
    const url = new URL(request.url);

    if (url.pathname === "/check-open-ai-key") {
      const hasOpenAIKey = !!process.env.OPENAI_API_KEY;
      return Response.json({
        success: hasOpenAIKey,
      });
    }
    if (!process.env.OPENAI_API_KEY) {
      console.error(
        "OPENAI_API_KEY is not set, don't forget to set it locally in .dev.vars, and use `wrangler secret bulk .dev.vars` to upload it to production"
      );
    }
    return (
      // Route the request to our agent or return 404 if not found
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  },
} satisfies ExportedHandler<Env>;
