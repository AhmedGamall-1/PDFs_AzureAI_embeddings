using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Embeddings;
using UglyToad.PdfPig;
using Milvus.Client;
using Azure;
using Azure.AI.OpenAI;
#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0011
#pragma warning disable SKEXP0010

namespace PdfChatDemo
{
    class Program
    {
        // Put your Azure OpenAI details here
        static string endpoint = "https://myproject-2025-resource.services.ai.azure.com/";
        static string apiKey = "5XHSMwg63zs3dW6HM9nXzajRK5oRqhdEHCAdVx6lxahfWXS4u6RcJQQJ99BHACfhMk5XJ3w3AAAAACOGseKC";
        static string chatModel = "gpt-4.1";
        static string embeddingModel = "text-embedding-ada-002";

        static MilvusClient? milvus;
        static string collectionName = "pdf_docsss";

        static async Task Main()
        {
            Console.WriteLine("=== PDF Chat Demo with Milvus ===\n");

            // Setup AI
            var kernel = Kernel.CreateBuilder()
                .AddAzureOpenAIChatCompletion(chatModel, endpoint, apiKey)
                .AddAzureOpenAITextEmbeddingGeneration(embeddingModel, endpoint, apiKey)
                .Build();

            var chat = kernel.GetRequiredService<IChatCompletionService>();
            var embedding = kernel.GetRequiredService<ITextEmbeddingGenerationService>();

            // Setup Milvus
            Console.WriteLine("Connecting to Milvus...");
            milvus = new MilvusClient("127.0.0.1", 19530,null,null,null);
            await SetupMilvus();
            Console.WriteLine("Milvus ready!\n");

            // Check if data already exists by doing a simple query
            var collection = milvus.GetCollection(collectionName);
            
            try
            {
                // Try to query for any data - if successful and returns results, data exists
                var testResults = await collection.QueryAsync("id > 0");  // Empty expression gets all data
                
                if (testResults.Any())
                {
                    Console.WriteLine("Found existing data. Skipping PDF processing.");
                }
                else
                {
                    Console.WriteLine("No existing data found. Processing PDF...");
                    await ProcessPdf();
                }
            }
            catch (Exception ex) 
            {
                // If query fails, assume no data exists
                Console.WriteLine($"Query failed: {ex.Message}");
                Console.WriteLine("No existing data found. Processing PDF...");

                await ProcessPdf();
            }
            
            async Task ProcessPdf()
            {
                // Get PDF
                Console.Write("Enter PDF file path: ");
                string pdfPath = "PDFs/Internal_Governance_and_Guidelines.pdf";

                if (!File.Exists(pdfPath))
                {
                    Console.WriteLine("PDF not found!");
                    return;
                }

                // Process PDF
                Console.WriteLine("Processing PDF...");
                string text = ReadPdf(pdfPath);
                var chunks = SplitText(text);
                
                Console.WriteLine($"Created {chunks.Count} chunks, storing in Milvus...");
                await StoreInMilvus(chunks, embedding);
            }
            
            Console.WriteLine("Ready to chat!\n");

            // Chat
            while (true)
            {
                Console.Write("You: ");
                string question = Console.ReadLine() ?? "";
                if (question.ToLower() == "quit") break;

                var answer = await GetAnswer(question, chat, embedding);
                Console.WriteLine($"Bot: {answer}");
            }
        }

        static async Task SetupMilvus()
        {
            // Check if collection exists, create only if it doesn't
            if (!await milvus!.HasCollectionAsync(collectionName))
            {
                Console.WriteLine("Creating new collection...");
                
                // Create collection
                var schema = new CollectionSchema
                {
                    Fields =
                    {
                        FieldSchema.Create<long>("id", isPrimaryKey: true, autoId: true),
                        FieldSchema.CreateVarchar("text", maxLength: 65535),
                        FieldSchema.CreateFloatVector("vector", 1536)
                    },
                    Name = collectionName,
                    Description = "PDF chunks"
                };

                var collection = await milvus.CreateCollectionAsync(collectionName, schema);

                // Create index
                await collection.CreateIndexAsync("vector", IndexType.IvfFlat, SimilarityMetricType.L2, 
                    new Dictionary<string, string> { { "nlist", "100" } });
                await collection.LoadAsync();
            }
            else
            {
                Console.WriteLine("Using existing collection...");
                var collection = milvus.GetCollection(collectionName);
                await collection.LoadAsync(); // Make sure it's loaded
            }
        }

        static string ReadPdf(string path)
        {
            var text = new StringBuilder();
            using var doc = PdfDocument.Open(path);
            
            foreach (var page in doc.GetPages())
            {
                text.AppendLine(page.Text);
            }
            
            
            return text.ToString();
        }

        static List<string> SplitText(string text)
        {
            var chunks = new List<string>();
            var sentences = text.Split('.', StringSplitOptions.RemoveEmptyEntries);
            var current = "";

            foreach (var sentence in sentences)
            {
                if (current.Length + sentence.Length > 800)
                {
                    if (!string.IsNullOrEmpty(current))
                    {
                        chunks.Add(current.Trim());
                        current = "";
                    }
                }
                current += sentence + ". ";
            }

            if (!string.IsNullOrEmpty(current))
                chunks.Add(current.Trim());

            return chunks;
        }

        static async Task StoreInMilvus(List<string> chunks, ITextEmbeddingGenerationService embedding)
        {
            var texts = new List<string>();
            var vectors = new List<ReadOnlyMemory<float>>();

            for (int i = 0; i < chunks.Count; i++)
            {
                var emb = await embedding.GenerateEmbeddingAsync(chunks[i]);

                texts.Add(chunks[i]);
                vectors.Add(emb);
            }

            var data = new FieldData[]
            {

                FieldData.CreateVarChar("text", texts),
                FieldData.CreateFloatVector("vector", vectors)
            };

            var collection = milvus!.GetCollection(collectionName);
            await collection.InsertAsync(data);
            Console.WriteLine("Data inserted successfully!");
        }

        static async Task<string> GetAnswer(string question, IChatCompletionService chat, ITextEmbeddingGenerationService embedding)
        {
            // Get question embedding
            var questionEmb = await embedding.GenerateEmbeddingAsync(question);

            // Search Milvus
            var collection = milvus!.GetCollection(collectionName);
            var searchParams = new SearchParameters
            {
              OutputFields = { "text" },  // Request the text field
                ConsistencyLevel = ConsistencyLevel.Strong,
                Parameters = { ["nprobe"] = "10" }
            };
            var results = await collection.SearchAsync(
            vectorFieldName: "vector",
            vectors: new[] { questionEmb },
            metricType: SimilarityMetricType.L2,
            limit: 5,
            parameters: searchParams 
            
            );
             var textFieldData = results.FieldsData.FirstOrDefault(f => f.FieldName == "text");
            var retrievedTexts = new List<string>();
            // Get relevant text
            var context = "";
            Console.WriteLine($"Found {results.Scores.Count} results");
            Console.WriteLine($"Best score: {results.Scores.FirstOrDefault()}");
            Console.WriteLine($"Context length: {context.Length}");
            Console.WriteLine($"Number of field data: {results.FieldsData.Count}");
            foreach (var field in results.FieldsData)
            {
             Console.WriteLine($"Field name: '{field.FieldName}', Type: {field.GetType()}");
            }

         if (textFieldData != null)
            {
                var stringData = ((FieldData<string>)textFieldData).Data;

                for (int i = 0; i < Math.Min(results.Scores.Count, stringData.Count); i++)
                {
                    var text = stringData[i];
                    context += text + "\n\n";
                }
            }
        Console.WriteLine($"Context length: {context.Length}");
            if (string.IsNullOrEmpty(context))
                return "No relevant information found.";

            // Get AI answer
            string prompt = $"Based on this PDF content:\n{context}\n\nAnswer: {question}";
            var response = await chat.GetChatMessageContentAsync(prompt);
            
            return response.Content ?? "Could not generate answer.";
        }
    }
}