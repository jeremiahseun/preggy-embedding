import { QdrantClient } from "@qdrant/js-client-rest";

import 'dotenv/config'; // To load the .env file
import type { EmbedProps } from "./utils/interfaces.js";
import { embedQueryCohere } from "./models/cohere.js";

// --- CONFIGURATION ---
const COLLECTION_NAME = process.env.COLLECTION_NAME!;
const VECTOR_DIMENSION = process.env.VECTOR_DIMENSION;


// Initialize the Qdrant client
const qdrantClient = new QdrantClient({
    url: process.env.QDRANT_URL || "http://localhost:6333",
});


export async function upsertToQdrant({ chunks, payloads, embeddings }: EmbedProps) {
    console.log("Saving vectors to Qdrant...");

    await qdrantClient.upsert(COLLECTION_NAME, {
        wait: true,
        batch: {
            ids: chunks.map(chunk => chunk.chunkId), // Use our generated IDs
            vectors: embeddings,
            payloads: payloads,
        },
    });
    console.log("✅ Successfully saved embeddings!");
}


export async function getCollection() {
    try {
        // FIRSTLY CHECK IF COLLECTION EXIST
        const collections = await qdrantClient.getCollections();
        const collectionExist = collections.collections.find((c) => c.name === COLLECTION_NAME);
        if (!collectionExist) {
            // Create collection
            console.log(`Creating collection: ${COLLECTION_NAME}`);
            const vectorSize = VECTOR_DIMENSION ? parseInt(VECTOR_DIMENSION) : 1024;
            await qdrantClient.createCollection(COLLECTION_NAME, {
                vectors: {
                    size: vectorSize,
                    distance: "Cosine", // Cosine is great for text similarity
                },
            });
        } else {
            console.log(`Collection Exists: ${COLLECTION_NAME}`);
        }
    } catch (error) {
        console.log("Unable to load collection", error)
    }
}

/**
 * Searches the knowledge base in Qdrant for a given user query.
 *
 * @param query The user's search query string.
 * @param limit The maximum number of results to return.
 * @returns A promise that resolves to the search results from Qdrant.
 */
export async function searchKnowledgeBase(query: string, limit: number = 3) {
    console.log(`\nSearching for: "${query}"`);

    // 1. Embed the user's query using the SAME model family (Cohere)
    const queryEmbedding = await embedQueryCohere(query);

    // Handle cases where embedding fails
    if (!queryEmbedding) {
        console.error("Could not generate query embedding. Aborting search.");
        return null;
    }

    // 2. Now, search Qdrant using the generated vector
    try {
        console.log("Searching Qdrant for the most relevant chunks...");
        const searchResult = await qdrantClient.search(COLLECTION_NAME, {
            vector: queryEmbedding,
            limit: limit, // Get the top N most relevant results
            with_payload: true, // Crucial to get the text and metadata back
        });

        console.log("\nTop relevant chunks found:");
        searchResult.forEach((result, i) => {
            console.log(`--- Result ${i + 1} (Score: ${result.score.toFixed(4)}) ---`);
            // The payload contains the clean, original text and other metadata
            console.log(result.payload?.text);
            console.log(`Source: ${result.payload?.source_url}`);
        });

        return searchResult;

    } catch (error) {
        console.error("Error searching Qdrant:", error instanceof Error ? error.message : error);
        return null;
    }
}
