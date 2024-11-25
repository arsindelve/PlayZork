﻿
using AdventurerEngine;
using ConsoleRunner;
using Engine.GameApiClient;
using OpenAI;

var adventurer = new Adventurer();
var session = "Bdet9wwwddeee65d890wetwegweeg000";
var gameClient = new ZorkApiClient();
var chatClient = new ChatGPTClient(ConsoleHelper.CreateLogger());

var history = new LimitedStack<(string, string)>(15);
var map = new UniqueLimitedStack<string>(25);
var items = new UniqueLimitedStack<string>(25);
var memory = new Memory(35);

var lastLocation = "West Of House";

chatClient.SystemPrompt = Prompts.SystemPrompt;

await gameClient.GetAsync(new ZorkApiRequest("verbose", session));
var lastResponse = await gameClient.GetAsync(new ZorkApiRequest("look", session));

if (lastResponse == null)
    throw new Exception("Null from Zork");

for (var i = 0; i < 250; i++)
{
    Console.ForegroundColor = ConsoleColor.White;
    Console.WriteLine(lastResponse.Response);

    Thread.Sleep(TimeSpan.FromSeconds(10));

    var itemString = Builders.BuildItems(items);
    var mapString = Builders.BuildMap(map);
    var historyString = Builders.BuildHistory(history);
    var memoryString = Builders.BuildMemory(memory);

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

    var chatResponse = await adventurer.ChatResponse(lastResponse, itemString, mapString, memoryString, historyString, chatClient);

    Console.ForegroundColor = ConsoleColor.DarkGreen;
    Console.WriteLine("> " + chatResponse.Command);

    Console.ForegroundColor = ConsoleColor.Blue;
    Console.WriteLine(chatResponse.Reason);

    if (!string.IsNullOrEmpty(chatResponse.Item) && !chatResponse.Item.Contains("none "))
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine(chatResponse.Item);
        items.Push(chatResponse.Item);
    }

    if (chatResponse.RememberImportance > 0)
    {
        memory.Push(chatResponse);

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"{chatResponse.RememberImportance}: {chatResponse.Remember}");
    }

    lastResponse = await gameClient.GetAsync(new ZorkApiRequest(chatResponse.Command, session));
    if (lastResponse == null)
        throw new Exception("Null from Zork");

    if (lastResponse.LocationName != lastLocation)
    {
        var locationReminder =
            $"From: {lastLocation} To: {lastResponse.LocationName} Direction: {chatResponse.Command}";
        map.Push(locationReminder);
        lastLocation = lastResponse.LocationName;
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine(locationReminder);
    }

    history.Push((lastResponse.Response, chatResponse.Command));
}


