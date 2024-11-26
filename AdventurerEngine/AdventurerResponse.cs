namespace AdventurerEngine;

public class AdventurerResponse
{
    public required string Command { get; set; }

    public string? Reason { get; set; }

    public string? Remember { get; set; }
    
    public string? Item { get; set; }

    public decimal? RememberImportance { get; set; }
    
    public string? Moved { get; set; }
}