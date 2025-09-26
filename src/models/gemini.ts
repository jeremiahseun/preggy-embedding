import { GoogleGenerativeAI } from "@google/generative-ai";
import type { ChunkPayload, EmbedProps, ProcessedArticle } from "../utils/interfaces.js";
import 'dotenv/config'; // To load the .env file

// --- CONFIGURATION ---
const GEMINI_EMBEDDING_MODEL = process.env.GEMINI_EMBEDDING_MODEL!;
// Gemini's embedding-001 model has 768 dimensions


// Initialize the Google Generative AI client
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

const embeddingModel = genAI.getGenerativeModel({ model: GEMINI_EMBEDDING_MODEL });

export async function embedProcessedArticleGemini(processedArticle: ProcessedArticle): Promise<EmbedProps | null> {
    // Get the chunks of the processed articles
    const chunks = processedArticle.chunks;

    // The text to be embedded now includes the context
    const textsToEmbed = chunks.map(chunk => chunk.textForEmbedding);
    console.log(`Embedding ${textsToEmbed.length} chunks for article "${processedArticle.title}"...`);

    // --- NEW, RATE-LIMITED EMBEDDING LOGIC ---
    // Pass your `embeddingModel` instance to the function
    const embeddings = await embedSequentially(textsToEmbed);

    // If there was an error and embeddings is empty, stop here.
    if (embeddings.length === 0) {
        console.error("Embedding failed, no vectors to save.");
        return null;
    }
    console.log(`Gotten the embeddings! The count for embeddings is ${embeddings.length}, while comparing with the count of chunks: ${chunks.length}`);

    // --- Your Qdrant Upsert Logic Here ---
    // The payload now contains rich, structured data, which is great for filtering!
    console.log("Preparing the payloads for Qdrant...");
    const payloads: ChunkPayload[] = chunks.map(chunk => ({
        text: chunk.originalText, // Store the clean, original text
        article_id: chunk.articleId,
        article_title: processedArticle.title,
        category: processedArticle.category,
        source_url: processedArticle.sourceURL,
    }));

    console.log(`✅ Chunks, Embeddings and Payloads prepared!`);

    return (
        {
            chunks,
            embeddings,
            payloads
        }
    )
}

// Helper function to pause execution for a specified time
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Embeds a list of texts one by one with a delay between each request to avoid rate limits.
 * @param textsToEmbed An array of strings to embed.
 * @param embeddingModel The Gemini embedding model instance.
 * @param delayMs The delay in milliseconds to wait between each API call.
 * @returns A promise that resolves to an array of all successful embeddings.
 */
export async function embedSequentially(
    textsToEmbed: string[],
    delayMs: number = 1100 // Default to a safe 1.1-second delay
): Promise<number[][]> {

    const allEmbeddings: number[][] = [];
    const totalChunks = textsToEmbed.length;

    // Use a standard for-loop to have access to the index for logging
    for (let i = 0; i < totalChunks; i++) {
        const text = textsToEmbed[i];

        console.log(`Embedding chunk ${i + 1} of ${totalChunks}...`);

        try {
            // Use embedContent for a single piece of text
            const result = await embeddingModel.embedContent({
                content: { parts: [{ text: text as string }], role: 'user' },
            });
            console.log(`The embed should work now for chunk ${i + 1} of ${totalChunks}`)

            const embedding = result.embedding.values;
            allEmbeddings.push(embedding);

            // Wait after each request, but not after the very last one
            if (i < totalChunks - 1) {
                console.log('Sleeping...')
                await sleep(delayMs);
            }

        } catch (error) {
            console.error(`Error embedding chunk ${i + 1}:`, error instanceof Error ? error.message : error);
            // We'll skip this chunk and continue with the next ones.
        }
    }

    console.log(`Successfully embedded ${allEmbeddings.length} of ${totalChunks} chunks.`);
    return allEmbeddings;
}
