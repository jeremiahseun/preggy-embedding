import type { ProcessedArticle } from "./utils/interfaces.js";
import { upsertToQdrant } from "./qdrant.js";
import { embedProcessedArticleCohere } from "./models/cohere.js";

export async function embedDocuments(processedArticle: ProcessedArticle) {
    // CHUNK AND EMBED THE DOCUMENT
    console.log("Start embedding workflow...");
    // CALL OPENAI TO EMBED CHUNKS AND RETURN
    // GEMINI LAYER
    const embeddingResult = await embedProcessedArticleCohere(processedArticle);
    // CONFIRM THE RESULT AND SEND TO QDRANT
    if (embeddingResult) {
        await upsertToQdrant({
            chunks: embeddingResult.chunks,
            embeddings: embeddingResult.embeddings,
            payloads: embeddingResult.payloads
        })
        console.log("We should be done by now")
    } else {
        console.log("Didn't work with the OpenAI part")
    }
}
