import { Kafka } from 'kafkajs';
import { searchArticles } from '../services/articleProcessor';
import { GoogleGenerativeAI } from '@google/generative-ai';

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

export async function startConsumer() {
  const consumer = kafka.consumer({
    groupId: `${process.env.KAFKA_GROUP_ID_PREFIX}${Date.now()}`
  });

  try {
    console.log('Connecting Kafka consumer...');
    await consumer.connect();
    console.log('Subscribed to topic:', process.env.KAFKA_TOPIC_NAME!);
    await consumer.subscribe({
      topic: process.env.KAFKA_TOPIC_NAME!,
      fromBeginning: false
    });

    await consumer.run({
      eachMessage: async ({ topic, partition, message }) => {
        const query = message.value?.toString();
        if (!query) {
          console.warn('Message without a valid query:', message);
          return;
        }

        console.log(`Processing query [Partition ${partition}]: ${query}`);

        try {
          // Perform RAG using Gemini
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

          console.log('Generated response:', response.text());
          console.log('Sources:', relevantArticles.map(a => ({
            title: a.title,
            url: a.url,
            date: a.date
          })));
        } catch (error) {
          console.error(`Failed to process query ${query}:`, error);
        }
      },
    });

    process.on('SIGINT', async () => {
      console.log('Shutting down consumer...');
      await consumer.disconnect();
      process.exit(0);
    });

  } catch (error) {
    console.error('Kafka consumer error:', error);
    process.exit(1);
  }
}