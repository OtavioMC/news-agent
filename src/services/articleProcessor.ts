import cheerio from 'cheerio';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Pinecone } from '@pinecone-database/pinecone';

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY!
});

interface Article {
    title: string;
    content: string;
    url: string;
    date: string;
}

interface ResponseWithSources {
    answer: string;
    sources: Article[];
}

export async function processArticle(url: string): Promise<Article> {
    try {
        // 1. Fetch HTML content
        const html = await fetch(url).then(res => res.text());
        if (!html || typeof html !== 'string') {
            throw new Error('Failed to fetch valid HTML content.');
        }

        // 2. Extract main content with Cheerio
        const $ = cheerio.load(html);
        const title = $('h1').first().text().trim() || $('title').text().trim();
        const paragraphs = $('p').map((_, el) => $(el).text().trim()).get();
        const rawContent = paragraphs.join('\n\n');

        // 3. Clean and structure with Gemini
        const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro-exp-03-25" });
        const prompt = `Clean and structure this news article into JSON format:
                        Title: ${title}
                        Content: ${rawContent.substring(0, 10000)}

                        Output format:
                        {
                            "title": "Clean article title",
                            "content": "Cleaned and concise article content",
                            "url": "${url}",
                            "date": "YYYY-MM-DD"
                        }`;

        const result = await model.generateContent(prompt);
        const text = result.response.text();
        const jsonStart = text.indexOf('{');
        const jsonEnd = text.lastIndexOf('}') + 1;
        return JSON.parse(text.slice(jsonStart, jsonEnd));
    } catch (error) {
        console.error(`Failed to process article ${url}:`, error);
        throw error;
    }
}

export async function generateResponse(query: string, articles: Article[]): Promise<ResponseWithSources> {
    try {
        const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro-exp-03-25" });

        // Prepare context with proper source references
        const context = articles.map(article => 
            `[${article.title}](${article.url}) - Accessed ${article.date || new Date().toISOString().split('T')[0]}\n` +
            `Content: ${article.content.substring(0, 1000)}`
        ).join('\n\n');

        const prompt = `Answer this query: ${query}\n\n` +
                      `Available Sources:\n${context}\n\n` +
                      `Format Requirements:\n` +
                      `1. Include ALL sources used in this exact format: "[Title](URL)"\n` +
                      `2. Place source references at relevant points in your answer\n` +
                      `3. List ALL used sources again at the end in a "Sources:" section`;

        const { response } = await model.generateContent(prompt);
        const answer = response.text();

        // Extract sources using more flexible regex
        const sourceRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
        const sources: Article[] = [];
        let match;
        const usedUrls = new Set<string>();

        // Find all source references in answer
        while ((match = sourceRegex.exec(answer)) !== null) {
            const url = match[2];
            if (!usedUrls.has(url)) {
                const article = articles.find(a => a.url === url);
                if (article) {
                    sources.push(article);
                    usedUrls.add(url);
                }
            }
        }

        return {
            answer: answer,
            sources: sources
        };
    } catch (error) {
        console.error('Failed to generate response:', error);
        throw error;
    }
}

async function storeArticle(article: Article): Promise<void> {
    try {
        // 1. Generate embeddings using Gemini's embedding model
        const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-exp-03-07" });
        const { embedding } = await embeddingModel.embedContent({
            content: {
                role: "user",
                parts: [{ text: `${article.title}\n${article.content.substring(0, 1000)}` }]
            }
        });

        // 2. Get or create Pinecone index
        const index = pinecone.Index(process.env.PINECONE_INDEX_NAME!);

        // 3. Create unique ID from URL
        const articleId = Buffer.from(article.url).toString('base64').slice(0, 64);

        // 4. Prepare metadata
        const metadata = {
            title: article.title.substring(0, 1000),
            url: article.url,
            date: article.date || new Date().toISOString().split('T')[0],
            content: article.content.substring(0, 5000)
        };

        // 5. Upsert the vector
        await index.upsert([{
            id: articleId,
            values: embedding.values,
            metadata
        }]);

        console.log(`Stored article: ${article.title.substring(0, 50)}...`);
    } catch (error) {
        console.error('Error storing article:', error);
        throw error;
    }
}

export async function searchArticles(query: string, topK = 3): Promise<Article[]> {
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