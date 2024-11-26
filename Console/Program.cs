using AdventurerEngine;
using ConsoleRunner;

var adventurer = await new Adventurer(ConsoleHelper.CreateLogger()).Initialize();

for (var i = 0; i < 250; i++)
{
    Console.ForegroundColor = ConsoleColor.White;
    Console.WriteLine(adventurer.LastZorkResponse?.Response);

    Thread.Sleep(TimeSpan.FromSeconds(10));

    var itemString = adventurer.ItemString;
    var mapString = adventurer.MapString;
    var historyString = adventurer.HistoryString;
    var memoryString = adventurer.MemoryString;

    // if (!string.IsNullOrEmpty(historyString))
    // {
    //     Console.ForegroundColor = ConsoleColor.Cyan;
    //     Console.WriteLine("------------------------------- History---------------------------------------");
    //     Console.WriteLine(historyString);
    // }
    //
    // if (!string.IsNullOrEmpty(itemString))
    // {
    //     Console.ForegroundColor = ConsoleColor.DarkRed;
    //     Console.WriteLine("------------------------------- Items---------------------------------------");
    //     Console.WriteLine(itemString);
    // }
    //
    // if (!string.IsNullOrEmpty(memoryString))
    // {
    //     Console.ForegroundColor = ConsoleColor.Yellow;
    //     Console.WriteLine("------------------------------- Memories---------------------------------------");
    //     Console.WriteLine(memoryString);
    // }
    //
    if (!string.IsNullOrEmpty(mapString))
    {
        Console.ForegroundColor = ConsoleColor.Blue;
        Console.WriteLine("------------------------------- Map ---------------------------------------");
        Console.WriteLine(mapString);
    }

    AdventurerResponse response = await adventurer.Play();

    Console.ForegroundColor = ConsoleColor.DarkGreen;
    Console.WriteLine("> " + response.Command);

    Console.ForegroundColor = ConsoleColor.Blue;
    Console.WriteLine(response.Reason);

    if (!string.IsNullOrEmpty(response.Item) && !response.Item.Contains("none "))
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine(response.Item);
    }

    if (response.RememberImportance > 0)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"{response.RememberImportance}: {response.Remember}");
    }
}