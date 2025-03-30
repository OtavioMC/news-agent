import { Kafka, KafkaJSError } from 'kafkajs';
import { GoogleGenerativeAI } from '@google/generative-ai';
import cheerio from 'cheerio';
import { Pinecone } from '@pinecone-database/pinecone';

const kafka = new Kafka({
  clientId: 'news-agent-consumer',
  brokers: [process.env.KAFKA_BROKER!],
  ssl: true,
  sasl: {
    mechanism: 'plain',
    username: process.env.KAFKA_USERNAME!,
    password: process.env.KAFKA_PASSWORD!,
  }
});

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });

interface Article {
  title: string;
  content: string;
  url: string;
  date: string;
}

export async function startConsumer() {
  const consumer = kafka.consumer({
    groupId: `${process.env.KAFKA_GROUP_ID_PREFIX}${Date.now()}`
  });

  try {
    console.log('Connecting Kafka consumer...');
    await consumer.connect();
    await consumer.subscribe({ 
      topic: process.env.KAFKA_TOPIC_NAME!,
      fromBeginning: false 
    });

    console.log(`Listening for article URLs on topic: ${process.env.KAFKA_TOPIC_NAME}`);

    await consumer.run({
      eachMessage: async ({ partition, message }) => {
        const url = message.value?.toString();
        if (!url || !url.startsWith('http')) {
          console.warn(`Invalid URL in message [Partition ${partition}]`);
          return;
        }

        console.log(`Processing article URL [Partition ${partition}]: ${url}`);

        try {
          // 1. Extract HTML content
          const html = await fetch(url).then(res => res.text());
          if (!html) throw new Error('Failed to fetch HTML content');
          
          // 2. Basic content extraction with Cheerio
          const $ = cheerio.load(html);
          const rawTitle = $('h1').first().text().trim() || $('title').text().trim();
          const rawContent = $('body').text().replace(/\s+/g, ' ').trim();

          // 3. Clean and structure with Gemini
          const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro-exp-03-25" });
          const prompt = `
            Transform this raw article content into structured JSON:
            URL: ${url}
            Raw Title: ${rawTitle.substring(0, 200)}
            Raw Content: ${rawContent.substring(0, 10000)}

            Output MUST be valid JSON in this exact format:
            {
              "title": "Cleaned article title",
              "content": "Cleaned and concise article content",
              "url": "${url}",
              "date": "YYYY-MM-DD" (use current date if not available)
            }`;

          const { response } = await model.generateContent(prompt);
          const text = response.text();
          const jsonStart = text.indexOf('{');
          const jsonEnd = text.lastIndexOf('}') + 1;
          const article: Article = JSON.parse(text.slice(jsonStart, jsonEnd));

          // 4. Store in Pinecone
          const embeddingModel = genAI.getGenerativeModel({ model: "gemini-embedding-exp-03-07" });
          const { embedding } = await embeddingModel.embedContent({
            content: { 
              role: "user",
              parts: [{ text: `${article.title}\n${article.content.substring(0, 1000)}` }]
            }
          });

          await pinecone.Index(process.env.PINECONE_INDEX_NAME!).upsert([{
            id: Buffer.from(url).toString('base64').slice(0, 64),
            values: embedding.values,
            metadata: article
          }]);

          console.log('Successfully processed article:', {
            title: article.title,
            url: article.url,
            date: article.date,
            contentLength: article.content.length
          });

        } catch (error: unknown) {
          console.error(`Failed to process article ${url}:`, 
            error instanceof Error ? error.message : error);
        }
      },
    });

    process.on('SIGINT', async () => {
      await consumer.disconnect();
      console.log('Consumer disconnected');
      process.exit(0);
    });

  } catch (error: unknown) {
    console.error('Consumer initialization failed:',
      error instanceof Error ? error.message : 'Unknown error');
    process.exit(1);
  }
}