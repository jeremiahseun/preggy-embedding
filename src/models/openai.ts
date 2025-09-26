import OpenAI from "openai";
import type { ChunkPayload, EmbedProps, ProcessedArticle } from "../utils/interfaces.js";
import 'dotenv/config'; // To load the .env file

// --- CONFIGURATION ---
const OPENAI_EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small";


// Initialize the OpenAI client
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY || "",
});

export async function embedProcessedArticleOpenAI(processedArticle: ProcessedArticle): Promise<EmbedProps | null> {
    // Get the chunks of the processed articles
    const chunks = processedArticle.chunks;

    // The text to be embedded now includes the context
    const textsToEmbed = chunks.map(chunk => chunk.textForEmbedding);
    console.log(`Embedding ${textsToEmbed.length} chunks for article "${processedArticle.title}" using OpenAI...`);

    // --- BATCH EMBEDDING LOGIC ---
    const embeddings = await embedInBatches(textsToEmbed);

    // If there was an error and embeddings is empty, stop here.
    if (embeddings.length === 0) {
        console.error("OpenAI Embedding failed, no vectors to save.");
        return null;
    }
    console.log(`Gotten the embeddings! The count for embeddings is ${embeddings.length}, while comparing with the count of chunks: ${chunks.length}`);

    // The payload now contains rich, structured data, which is great for filtering!
    console.log("Preparing the payloads for Qdrant...");
    const payloads: ChunkPayload[] = chunks.map(chunk => ({
        text: chunk.originalText, // Store the clean, original text
        article_id: chunk.articleId,
        article_title: processedArticle.title,
        category: processedArticle.category,
        source_url: processedArticle.sourceURL,
    }));

    console.log(`✅ Chunks, Embeddings and Payloads prepared for OpenAI!`);

    return {
        chunks,
        embeddings,
        payloads
    };
}

/**
 * Embeds a list of texts in batches using the OpenAI API.
 * @param textsToEmbed An array of strings to embed.
 * @param batchSize The number of texts to embed in each API call.
 * @returns A promise that resolves to an array of all successful embeddings.
 */
export async function embedInBatches(
    textsToEmbed: string[],
    batchSize: number = 50 // A reasonable batch size
): Promise<number[][]> {
    const allEmbeddings: number[][] = [];
    const totalChunks = textsToEmbed.length;

    for (let i = 0; i < totalChunks; i += batchSize) {
        const batch = textsToEmbed.slice(i, i + batchSize);
        const start = i + 1;
        const end = Math.min(i + batchSize, totalChunks);

        console.log(`Embedding batch ${start}-${end} of ${totalChunks}...`);

        try {
            const response = await openai.embeddings.create({
                model: OPENAI_EMBEDDING_MODEL,
                input: batch,
                encoding_format: "float",
                dimensions: 512
            });

            const embeddings = response.data.map((item) => item.embedding);
            allEmbeddings.push(...embeddings);

        } catch (error) {
            console.error(`Error embedding batch ${start}-${end}:`, error instanceof Error ? error.message : error);
            // Skip this batch and continue with the next ones.
            // You might want to implement a retry mechanism here.
        }
    }

    console.log(`Successfully embedded ${allEmbeddings.length} of ${totalChunks} chunks with OpenAI.`);
    return allEmbeddings;
}
