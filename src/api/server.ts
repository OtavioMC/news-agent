// src/api/server.ts
import express, { Express, Request, Response, NextFunction } from 'express';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Pinecone } from '@pinecone-database/pinecone';
import cheerio from 'cheerio';
import dotenv from 'dotenv';
dotenv.config();

// Initialize Express app
const app: Express = express();
app.use(express.json());

// Initialize AI and DB clients
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });

interface Article {
  title: string;
  content: string;
  url: string;
  date: string;
}

interface RequestBody {
  query: string;
}

// Utility function to extract URL from query
function extractUrlFromQuery(query: string): string | null {
  const urlRegex = /https?:\/\/[^\s]+/;
  const match = query.match(urlRegex);
  return match ? match[0] : null;
}

// Article processing function
async function processArticle(url: string): Promise<Article> {
  try {
    const html = await fetch(url).then(res => res.text());
    if (!html || typeof html !== 'string') {
      throw new Error('Failed to fetch valid HTML content.');
    }

    const $ = cheerio.load(html);
    const title = $('h1').first().text().trim();
    const content = $('p').map((_, el) => $(el).text().trim()).get().join('\n\n');

    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });
    const prompt = `Extract JSON from this article:\nTitle: ${title}\nContent: ${content.substring(0, 10000)}\n\nOutput format: {
      "title": "string",
      "content": "string",
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

// Search function using Pinecone
async function searchArticles(query: string, topK = 3): Promise<Article[]> {
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

// API Endpoint handler function
const agentHandler = async (req: Request<{}, {}, RequestBody>, res: Response, next: NextFunction) => {
  try {
    const { query } = req.body;

    if (!query || typeof query !== 'string') {
      return res.status(400).json({ error: 'Query is required and must be a string' });
    }

    const url = extractUrlFromQuery(query);
    if (url) {
      const article = await processArticle(url);
      const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });
      const { response } = await model.generateContent(
        `Summarize this article in 3-4 sentences:\n\n${article.content}`
      );

      return res.json({
        answer: response.text(),
        sources: [{
          title: article.title,
          url: article.url,
          date: article.date
        }]
      });
    }

    const relevantArticles = await searchArticles(query);
    const context = relevantArticles.map(a =>
      `Title: ${a.title}\nContent: ${a.content.substring(0, 1000)}`
    ).join('\n\n');

    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });
    const { response } = await model.generateContent({
      contents: [{
        role: "user",
        parts: [{
          text: `Answer this query: ${query}\n\nContext:\n${context}\n\nBe concise and cite sources.`
        }]
      }]
    });

    return res.json({
      answer: response.text(),
      sources: relevantArticles.map(a => ({
        title: a.title,
        url: a.url,
        date: a.date
      }))
    });

  } catch (error) {
    console.error('API Error:', error);
    return res.status(500).json({
      error: 'Processing failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
};

// Register the route handler
app.post('/agent', (req: Request<{}, {}, RequestBody>, res: Response, next: NextFunction) => {
  agentHandler(req, res, next).catch(next);
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

export default app;