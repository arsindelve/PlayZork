using System.Diagnostics;
using AdventurerEngine.DataStructure;
using Engine.GameApiClient;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using OpenAI;

namespace AdventurerEngine;

public class Adventurer(ILogger logger)
{
    private const string Session = "XXXXXXXXXX";

    private readonly ChatGPTClient _chatClient = new(logger)
    {
        SystemPrompt = Prompts.SystemPrompt
    };

    private readonly ZorkApiClient _gameClient = new();
    private readonly LimitedStack<(string, string)> _history = new(15);
    private readonly UniqueLimitedStack<string> _items = new(25);
    private readonly Map _map = new(50);
    private readonly Memory _memory = new(35);
    private string? _lastCommand;

    public ZorkApiResponse? LastZorkResponse { get; private set; }

    public string ItemString => Builders.BuildItems(_items);
    public string MapString => Builders.BuildMap(_map);
    public string HistoryString => Builders.BuildHistory(_history);
    public string MemoryString => Builders.BuildMemory(_memory);

    public async Task<Adventurer> Initialize()
    {
        await _gameClient.GetAsync(new ZorkApiRequest("verbose", Session));
        LastZorkResponse = await _gameClient.GetAsync(new ZorkApiRequest("look", Session));

        if (LastZorkResponse == null)
            throw new Exception("Null from Zork");

        return this;
    }

    public async Task<AdventurerResponse> Play()
    {
        var chatResponse = await GetAdventurerRequest(LastZorkResponse);
        LastZorkResponse = await _gameClient.GetAsync(new ZorkApiRequest(chatResponse.Command, Session));

        _lastCommand = chatResponse.Command;

        if (LastZorkResponse == null)
            throw new Exception("Null from Zork");

        return chatResponse;
    }

    private async Task<AdventurerResponse> GetAdventurerRequest(ZorkApiResponse? zorkApiResponse)
    {
        if (zorkApiResponse == null)
            throw new Exception("Null from Zork");

        var request = new Request
        {
            UserMessage = Prompts.TurnPrompt(zorkApiResponse, MapString, ItemString, MemoryString, HistoryString)
        };

        var rawResponse = await _chatClient.CompleteChat(request);
        var gameResponse = ProcessResponse(zorkApiResponse, rawResponse);

        _memory.Degrade();

        return gameResponse;
    }

    private AdventurerResponse ProcessResponse(ZorkApiResponse zorkApiResponse, string rawResponse)
    {
        Debug.WriteLine(rawResponse);
        var gameResponse = JsonConvert.DeserializeObject<AdventurerResponse>(rawResponse);

        if (gameResponse is null)
            throw new Exception("Null from chat");

        if (gameResponse.Command is null)
            throw new Exception("Null command from chat");

        if (!string.IsNullOrEmpty(_lastCommand))
            _history.Push((zorkApiResponse.Response, _lastCommand));

        if (!string.IsNullOrEmpty(gameResponse.Item) && !gameResponse.Item.Contains("none "))
            _items.Push(gameResponse.Item);

        if (gameResponse.RememberImportance > 0)
            _memory.Push(gameResponse);

        ProcessMovement(zorkApiResponse);

        return gameResponse;
    }

    private void ProcessMovement(ZorkApiResponse zorkApiResponse)
    {
        if (string.IsNullOrEmpty(zorkApiResponse.LastMovementDirection))
            return;

        if (string.IsNullOrEmpty(zorkApiResponse.PreviousLocationName))
            return;

        if (string.IsNullOrEmpty(zorkApiResponse.LocationName))
            return;

        if (zorkApiResponse.LocationName != zorkApiResponse.PreviousLocationName)
            _map.Push((zorkApiResponse.PreviousLocationName, zorkApiResponse.LastMovementDirection,
                zorkApiResponse.LocationName));
        else
            _map.Push((zorkApiResponse.PreviousLocationName, zorkApiResponse.LastMovementDirection, null));
    }
}