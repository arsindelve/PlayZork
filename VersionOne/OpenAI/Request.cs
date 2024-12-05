namespace OpenAI;

public record Request
{
    public virtual string? UserMessage { get; init; }
}