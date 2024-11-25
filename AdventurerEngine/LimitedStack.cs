namespace AdventurerEngine;

/// <summary>
///     Represents a limited-size stack - once you have the maximum number of items, pushing another
///     item deletes the oldest.
/// </summary>
/// <typeparam name="T">The type of elements in the stack.</typeparam>
public class LimitedStack<T>(int size)
{
    protected readonly LinkedList<T> List = new();

    public virtual void Push(T item)
    {
        List.AddLast(item);
        if (List.Count > size) List.RemoveFirst();
    }

    public List<T> GetAll()
    {
        return List.ToList();
    }

    public T? Peek()
    {
        if (List.Count == 0)
            return default;

        return List.Last == null ? default : List.Last.Value;
    }
}