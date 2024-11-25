namespace AdventurerEngine;

public class UniqueLimitedStack<T>(int size) : LimitedStack<T>(size)
{
    public override void Push(T item)
    {
        if (List.Contains(item))
            return;
        
        base.Push(item);
    }
}