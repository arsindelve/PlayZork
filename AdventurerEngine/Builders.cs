﻿using System.Text;

namespace AdventurerEngine;

public static class Builders
{
    
    public static string BuildMap(UniqueLimitedStack<string> uniqueLimitedStack)
    {
        var stringBuilder = new StringBuilder();
        foreach (var line in uniqueLimitedStack.GetAll())
        {
            stringBuilder.AppendLine(line);
        }

        return stringBuilder.ToString();
    }
    
    public static string BuildMemory(Memory memory1)
    {
        var stringBuilder = new StringBuilder();
        foreach (var item in memory1.GetAll().OrderByDescending(item => item.RememberImportance.GetValueOrDefault()))
            stringBuilder.AppendLine($"Importance: {item.RememberImportance}. Reminder: {item.Remember}");

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