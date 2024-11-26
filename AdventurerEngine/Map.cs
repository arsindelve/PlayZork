namespace AdventurerEngine;

public class Map(int size) : LimitedStack<(string, string, string?)>(size)
{
    public override void Push((string, string, string?) item)
    {
        (string, string, string?)? existingEntry = List.FirstOrDefault(x => x.Item1 == item.Item1 && x.Item2 == item.Item2);

        if (existingEntry is not null)
            List.Remove(existingEntry.GetValueOrDefault());
        
        base.Push(item);
    }
}