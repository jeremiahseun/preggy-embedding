/**
 * NOTE: You will need to install the Cohere TypeScript SDK first:
 * npm install cohere-ai
 */
import { CohereClient } from "cohere-ai";
import type { ChunkPayload, EmbedProps, ProcessedArticle } from "../utils/interfaces.js"; // Assuming these types exist
import 'dotenv/config'; // To load the .env file

// --- CONFIGURATION ---
// The recommended model for high-quality retrieval.
const COHERE_EMBEDDING_MODEL = process.env.COHERE_EMBEDDING_MODEL || "embed-english-v3.0";

// Initialize the Cohere client using your Trial API Key
const cohere = new CohereClient({
    token: process.env.COHERE_API_KEY || "",
});

/**
 * Processes a single article, embeds its chunks using the Cohere API,
 * and prepares the data for upserting into a vector database.
 * @param processedArticle The article data processed into chunks.
 * @returns A promise that resolves to an EmbedProps object or null on failure.
 */
export async function embedProcessedArticleCohere(processedArticle: ProcessedArticle): Promise<EmbedProps | null> {
    // Get the chunks of the processed articles
    const chunks = processedArticle.chunks;

    // The text to be embedded now includes the context
    const textsToEmbed = chunks.map(chunk => chunk.textForEmbedding);
    console.log(`Embedding ${textsToEmbed.length} chunks for article "${processedArticle.title}" using Cohere...`);

    // --- BATCH EMBEDDING LOGIC ---
    const embeddings = await embedInBatchesCohere(textsToEmbed);

    // If there was an error and embeddings is empty, stop here.
    if (embeddings.length === 0) {
        console.error("Cohere Embedding failed, no vectors to save.");
        return null;
    }
    console.log(`Gotten the embeddings! The count for embeddings is ${embeddings.length}, while comparing with the count of chunks: ${chunks.length}`);

    // The payload now contains rich, structured data for filtering in Qdrant
    console.log("Preparing the payloads for Qdrant...");
    const payloads: ChunkPayload[] = chunks.map(chunk => ({
        text: chunk.originalText, // Store the clean, original text
        article_id: chunk.articleId,
        article_title: processedArticle.title,
        category: processedArticle.category,
        source_url: processedArticle.sourceURL,
    }));

    console.log(`✅ Chunks, Embeddings and Payloads prepared for Cohere!`);

    return {
        chunks,
        embeddings,
        payloads
    };
}

/**
 * Embeds a list of texts in batches using the Cohere API.
 * @param textsToEmbed An array of strings to embed.
 * @param batchSize The number of texts to embed in each API call. Cohere's max is 96.
 * @returns A promise that resolves to an array of all successful embeddings.
 */
export async function embedInBatchesCohere(
    textsToEmbed: string[],
    batchSize: number = 96 // Use Cohere's maximum allowed batch size for efficiency
): Promise<number[][]> {
    const allEmbeddings: number[][] = [];
    const totalChunks = textsToEmbed.length;

    for (let i = 0; i < totalChunks; i += batchSize) {
        const batch = textsToEmbed.slice(i, i + batchSize);
        const start = i + 1;
        const end = Math.min(i + batchSize, totalChunks);

        console.log(`Embedding batch ${start}-${end} of ${totalChunks}...`);

        try {
            // Call the Cohere embed API
            const response = await cohere.v2.embed({
                texts: batch,
                model: COHERE_EMBEDDING_MODEL,
                /**
                 * CRITICAL PARAMETER: 'input_type' tells Cohere the intended use.
                 * Use 'search_document' for the text you store in the vector DB.
                 * Later, when you embed a user's query, you will use 'search_query'.
                 * This optimizes the embeddings for relevance.
                 */
                inputType: "search_document",
                embeddingTypes: ['float']
                /**
                 * Unlike OpenAI's text-embedding-3 models, Cohere's models have a fixed
                 * dimension size. For 'embed-english-v3.0', it is 1024.
                 * There is no 'dimensions' parameter to set.
                 */
            });

            // The response object directly contains the embeddings array
            const embeddings = response.embeddings.float;
            if (embeddings) {
                allEmbeddings.push(...embeddings);
            } else {
                console.error(`No embeddings returned for batch ${start}-${end}.`);
            }

        } catch (error) {
            console.error(`Error embedding batch ${start}-${end}:`, error instanceof Error ? error.message : error);
            // In a production scenario, you might want to implement a retry mechanism here.
        }
    }

    console.log(`Successfully embedded ${allEmbeddings.length} of ${totalChunks} chunks with Cohere.`);
    return allEmbeddings;
}
