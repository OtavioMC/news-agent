import cheerio from 'cheerio';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Pinecone } from '@pinecone-database/pinecone';

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY!
});

export async function processArticle(url: string) {
    try {
        // 1. Fetch HTML content
        const html = await fetch(url).then(res => res.text());
        if (!html || typeof html !== 'string') {
            throw new Error('Failed to fetch valid HTML content.');
        }

        // 2. Extract main content with Cheerio
        const $ = cheerio.load(html);
        const title = $('h1').text().trim();
        const paragraphs = $('p').map((_, el) => $(el).text().trim()).get();
        const rawContent = paragraphs.join('\n\n');

        // 3. Clean and structure with Gemini
        const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro-exp-03-25" });
        const prompt = `
                        Clean and structure this news article into JSON format:
                        Title: ${title}
                        Content: ${rawContent.substring(0, 10000)}

                        Output format:
                        {
                            "title": "Clean article title",
                            "content": "Cleaned and concise article content",
                            "url": "${url}",
                            "date": "Extracted or current date in YYYY-MM-DD format"
                        }
                        `;

        let result;
        try {
            result = await model.generateContent(prompt);
        } catch (error) {
            console.error('Error generating content with Gemini:', error);
            throw new Error('Failed to process article content.');
        }

        let article;
        try {
            article = JSON.parse(result.response.text());
        } catch (error) {
            console.error('Error parsing Gemini response:', error);
            throw new Error('Invalid response format from Gemini.');
        }

        // 4. Generate embeddings and store in Pinecone
        await storeArticle(article);
    } catch (error) {
        console.error('Error processing article:', error);
        throw error;
    }
}

async function storeArticle(article: any) {
    try {
        // 1. Generate embeddings using Gemini's embedding model
        const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-exp-03-07" });
        let embeddingResult;
        try {
            embeddingResult = await embeddingModel.embedContent({
                content: {
                    role: "user",  // Required field
                    parts: [{
                        text: `${article.title}\n${article.content.substring(0, 1000)}`
                    }]
                }
            });
        } catch (error) {
            console.error('Error generating embeddings with Gemini:', error);
            throw new Error('Failed to generate embeddings.');
        }

        const embedding = embeddingResult.embedding.values;

        // 2. Get or create Pinecone index
        const indexName = process.env.PINECONE_INDEX_NAME || 'news-articles';
        const index = pinecone.Index(indexName);

        // 3. Create unique ID from URL (base64 encoded)
        const articleId = Buffer.from(article.url).toString('base64').slice(0, 64);

        // 4. Prepare metadata (Pinecone has 40KB limit per vector)
        const metadata = {
            title: article.title.substring(0, 1000), // Truncate if needed
            url: article.url,
            date: article.date || new Date().toISOString().split('T')[0],
            content: article.content.substring(0, 5000) // Store truncated content
        };

        // 5. Upsert the vector
        await index.upsert([{
            id: articleId,
            values: embedding,
            metadata: metadata
        }]);

        console.log(`Stored article: ${article.title}`);
    } catch (error) {
        console.error('Error storing article in Pinecone:', error);
        throw error;
    }
}

export async function searchArticles(query: string, topK = 3) {
    try {
        const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-exp-03-07" });
        const { embedding } = await embeddingModel.embedContent({
            content: { role: "user", parts: [{ text: query }] }
        });

        const results = await pinecone.Index(process.env.PINECONE_INDEX_NAME!).query({
            vector: embedding.values,
            topK,
            includeMetadata: true
        });

        return results.matches.map(match => ({
            title: match.metadata?.title as string,
            content: match.metadata?.content as string,
            url: match.metadata?.url as string,
            date: match.metadata?.date as string
        }));
    } catch (error) {
        console.error('Search failed:', error);
        throw error;
    }
}