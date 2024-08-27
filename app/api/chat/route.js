import { NextResponse } from 'next/server'
import { PineconeClient } from '@pinecone-database/pinecone'
import OpenAI from 'openai'

// System prompt
const systemPrompt = `
You are a rate my professor agent to help students find classes. For every user question, the top 3 professors that match the user question are returned. Use them to answer the question if needed.
`

export async function POST(req) {
  const data = await req.json()

  // Initialize Pinecone client
  const pc = new PineconeClient()
  await pc.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: 'us-east1-gcp' // Update this based on your region
  })
  const index = pc.Index('rag')

  // Initialize OpenAI client
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  })

  const text = data[data.length - 1].content

  // Create embedding for the input text
  const embeddingResponse = await openai.embeddings.create({
    model: 'text-embedding-ada-002',
    input: text,
  })

  const embedding = embeddingResponse.data[0].embedding

  // Query Pinecone index
  const results = await index.query({
    topK: 5,
    includeMetadata: true,
    vector: embedding,
    namespace: 'ns1',
  })

  // Process the results for the response
  let resultString = ''
  results.matches.forEach((match) => {
    resultString += `
    Returned Results:
    Professor: ${match.id}
    Review: ${match.metadata.review}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n`
  })

  // Update the user message with the results
  const lastMessage = data[data.length - 1]
  const lastMessageContent = lastMessage.content + resultString
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

  // Generate chat completion with OpenAI
  const completion = await openai.chat.completions.create({
    messages: [
      { role: 'system', content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: 'user', content: lastMessageContent },
    ],
    model: 'gpt-3.5-turbo',
    stream: true,
  })

  // Stream the response back to the client
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content
          if (content) {
            controller.enqueue(encoder.encode(content))
          }
        }
      } catch (err) {
        controller.error(err)
      } finally {
        controller.close()
      }
    },
  })

  return new NextResponse(stream)
}
