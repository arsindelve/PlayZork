using System.Diagnostics;
using Microsoft.Extensions.Logging;
using OpenAI;

namespace ConsoleRunner;

internal static class ConsoleHelper
{
    internal static ILogger CreateLogger()
    {
        ILoggerFactory loggerFactory;

        if (Debugger.IsAttached)
            loggerFactory = LoggerFactory.Create(builder =>
                builder
                    .AddConsole()
                    .AddDebug()
                    .AddFilter((category, _) =>
                    {
                        if (category!.Contains("GameEngine.GameEngine"))
                            return true;

                        return false;
                    })
                    .SetMinimumLevel(LogLevel.Debug)
            );
        else
            loggerFactory = LoggerFactory.Create(builder =>
                builder
                    .AddDebug()
                    .AddFilter((category, _) =>
                    {
                        if (category!.Contains("GameEngine.GameEngine"))
                            return true;

                        return false;
                    })
                    .SetMinimumLevel(LogLevel.Warning)
            );

        var logger = loggerFactory.CreateLogger<ChatGPTClient>();
        return logger;
    }
}