import type { RawArticle, ProcessedArticle, ArticleChunk } from "./utils/interfaces.js";
import { v4 as uuidv4 } from 'uuid';

/**
 * Processes a raw article into a structured format with enriched, semantically-aware chunks ready for embedding.
 *
 * @param article The raw article object.
 * @param chunkSize The TARGET maximum size of each text chunk (in characters). The final chunk may be slightly larger to respect sentence boundaries.
 * @param overlap The number of characters to overlap between chunks to maintain context.
 * @returns A ProcessedArticle object.
 */
export default function processArticleForEmbedding(
    article: RawArticle,
    chunkSize: number = 1024, // Recommended: ~256 tokens
    overlap: number = 200   // Recommended: ~50 tokens
): ProcessedArticle {
    console.log(`Processing article: "${article.title}"...`);

    const articleId = uuidv4();
    const processedChunks: ArticleChunk[] = [];
    const text = article.content;

    // --- NEW: SEMANTIC CHUNKING LOGIC ---
    let i = 0;
    while (i < text.length) {
        // 1. Define the ideal end point based on chunk size
        let idealEnd = i + chunkSize;
        let actualEnd = idealEnd;

        // 2. If the ideal end isn't the end of the whole text, find a better boundary
        if (idealEnd < text.length) {
            // Find the last sentence or paragraph break before the ideal end point
            const lastPeriod = text.lastIndexOf('.', idealEnd);
            const lastQuestionMark = text.lastIndexOf('?', idealEnd);
            const lastExclamation = text.lastIndexOf('!', idealEnd);
            const lastNewline = text.lastIndexOf('\n', idealEnd);

            // Use the latest of these natural breaking points as the actual end
            const bestEnd = Math.max(lastPeriod, lastQuestionMark, lastExclamation, lastNewline);

            // Use the natural break if it's found after the start of the current chunk
            if (bestEnd > i) {
                actualEnd = bestEnd + 1; // +1 to include the punctuation mark itself
            }
        }

        // 3. Slice the text to get the final chunk
        const originalChunkText = text.slice(i, actualEnd).trim();

        // If the chunk is empty (can happen with consecutive newlines), skip it
        if (!originalChunkText) {
            i += chunkSize - overlap;
            continue;
        }

        // 4. Create the context-rich string for the embedding model.
        const textForEmbedding = `Region: ${article.region}\nCategory: ${article.category}\nTitle: ${article.title}\n\n${originalChunkText}`;

        processedChunks.push({
            chunkId: uuidv4(),
            articleId: articleId,
            textForEmbedding: textForEmbedding,
            originalText: originalChunkText,
        });

        // 5. Move the starting point for the next chunk
        i += chunkSize - overlap;
    }

    console.log(`Article chunked into ${processedChunks.length} semantically-aware pieces.`);

    return {
        articleId,
        title: article.title,
        category: article.category,
        sourceURL: article.sourceURL ?? "",
        region: article.region ?? "",
        chunks: processedChunks,
    };
}
