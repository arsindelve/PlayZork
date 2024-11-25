
using AdventurerEngine;
using ConsoleRunner;
using Engine.GameApiClient;
using OpenAI;

var adventurer = new Adventurer();
var session = "Bdet77gg9wwwddeee65d890wetwegweeg000";
var gameClient = new ZorkApiClient();
var chatClient = new ChatGPTClient(ConsoleHelper.CreateLogger())
{
    SystemPrompt = Prompts.SystemPrompt
};

await gameClient.GetAsync(new ZorkApiRequest("verbose", session));
var lastResponse = await gameClient.GetAsync(new ZorkApiRequest("look", session));

if (lastResponse == null)
    throw new Exception("Null from Zork");

for (var i = 0; i < 250; i++)
{
    Console.ForegroundColor = ConsoleColor.White;
    Console.WriteLine(lastResponse.Response);

    Thread.Sleep(TimeSpan.FromSeconds(10));

    var itemString = adventurer.ItemString;
    var mapString = adventurer.MapString;
    var historyString = adventurer.HistoryString;
    var memoryString = adventurer.MemoryString;

    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine("------------------------------- History---------------------------------------");
    Console.WriteLine(historyString);

    Console.ForegroundColor = ConsoleColor.DarkRed;
    Console.WriteLine("------------------------------- Items---------------------------------------");
    Console.WriteLine(itemString);

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("------------------------------- Memories---------------------------------------");
    Console.WriteLine(memoryString);

    Console.ForegroundColor = ConsoleColor.Blue;
    Console.WriteLine("------------------------------- Map ---------------------------------------");
    Console.WriteLine(mapString);

    AdventurerResponse chatResponse = await adventurer.ChatResponse(lastResponse, chatClient);

    Console.ForegroundColor = ConsoleColor.DarkGreen;
    Console.WriteLine("> " + chatResponse.Command);

    Console.ForegroundColor = ConsoleColor.Blue;
    Console.WriteLine(chatResponse.Reason);

    if (!string.IsNullOrEmpty(chatResponse.Item) && !chatResponse.Item.Contains("none "))
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine(chatResponse.Item);
    }

    if (chatResponse.RememberImportance > 0)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"{chatResponse.RememberImportance}: {chatResponse.Remember}");
    }

    lastResponse = await gameClient.GetAsync(new ZorkApiRequest(chatResponse.Command, session));
    
    if (lastResponse == null)
        throw new Exception("Null from Zork");
    
}


