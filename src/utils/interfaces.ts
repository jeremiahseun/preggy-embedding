/**
 * Represents a raw article before it's processed for embedding.
 */
interface RawArticle {
    title: string;
    content: string;
    category: string;
    region?: string;
    sourceURL?: string; // Optional: The URL or source of the article
}

/**
 * Represents a single chunk of an article, prepared for embedding.
 */
interface ArticleChunk {
    chunkId: string; // Unique ID for the chunk
    articleId: string; // ID of the parent article
    textForEmbedding: string; // The text that will be sent to the Gemini API (with context)
    originalText: string; // The raw, original text of the chunk
    // You can add more metadata here if needed
}

/**
 * Represents a fully processed article, containing all its embeddable chunks.
 */
interface ProcessedArticle {
    articleId: string;
    title: string;
    category: string;
    region?: string;
    sourceURL?: string;
    chunks: ArticleChunk[];
}


type EmbedProps = {
    chunks: ArticleChunk[]
    embeddings: number[][]
    payloads: ChunkPayload[]
}

type ChunkPayload = {
    text: string;
    article_id: string;
    article_title: string;
    category: string;
    source_url: string | undefined;
}


export type { RawArticle, ArticleChunk, ProcessedArticle, EmbedProps, ChunkPayload }
