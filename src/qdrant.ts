import { QdrantClient } from "@qdrant/js-client-rest";

import 'dotenv/config'; // To load the .env file
import type { EmbedProps } from "./utils/interfaces.js";

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


async function search(query: string) {

    // // PERFORM A SEARCH (RETRIEVAL)
    // const userQuery = query;
    // console.log(`\nSearching for: "${userQuery}"`);

    // // First, embed the user's query using the SAME model
    // const queryEmbedding = (await embeddingModel.embedContent(userQuery)).embedding.values;

    // // Now, search Qdrant
    // const searchResult = await qdrantClient.search(COLLECTION_NAME, {
    //     vector: queryEmbedding,
    //     limit: 3, // Get the top 3 most relevant results
    // });

    // console.log("\nTop 3 relevant chunks found:");
    // searchResult.forEach((result, i) => {
    //     console.log(`--- Result ${i + 1} (Score: ${result.score.toFixed(4)}) ---`);
    //     console.log(result.payload?.text);
    // });
}
