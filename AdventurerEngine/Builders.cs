using System.Text;

namespace AdventurerEngine;

public static class Builders
{
    
    public static string BuildMap(Map map)
    {
        var stringBuilder = new StringBuilder();
        foreach (var line in map.GetAll())
        {
            string destination = string.IsNullOrEmpty(line.Item3) ? "Not possible" : $"Goes to: {line.Item3}";
            stringBuilder.AppendLine($"From {line.Item1}, Direction {line.Item2}, {destination}  ");
        }

        return stringBuilder.ToString();
    }
    
    public static string BuildMemory(Memory memory1)
    {
        var stringBuilder = new StringBuilder();
        foreach (var item in memory1.GetAll().OrderByDescending(item => item.RememberImportance))
            stringBuilder.AppendLine($"Importance: { Math.Round(item.RememberImportance.GetValueOrDefault(),0)}. Reminder: {item.Remember}");

        return stringBuilder.ToString();
    }
    
    public static string BuildItems(UniqueLimitedStack<string> uniqueLimitedStack)
    {
        int counter = 1;
        var stringBuilder = new StringBuilder();
        foreach (var item in uniqueLimitedStack.GetAll())
        {
            stringBuilder.AppendLine($"{counter}) {item}");
            counter++;
        }

        return stringBuilder.ToString();
    }
    
    public static string BuildHistory(LimitedStack<(string, string)> limitedStack)
    {
        var historyString = new StringBuilder();
        var counter = 1;
        var valueTuples = limitedStack.GetAll().ToList();
        valueTuples.Reverse();

        foreach (var item in valueTuples)
        {
            historyString.AppendLine($"{counter}) Your Request: {item.Item2} Server Response: {item.Item1.TrimEnd()}");
            counter++;
        }

        return historyString.ToString();
    }
}