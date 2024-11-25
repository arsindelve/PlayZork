namespace AdventurerEngine;

/// <summary>
///     Represents a limited-size stack - once you have the maximum number of items, pushing another
///     item deletes the oldest.
/// </summary>
/// <typeparam name="T">The type of elements in the stack.</typeparam>
public class LimitedStack<T>(int size)
{
    protected readonly LinkedList<T> _list = new();

    public virtual void Push(T item)
    {
        _list.AddLast(item);
        if (_list.Count > size) _list.RemoveFirst();
    }

    public List<T> GetAll()
    {
        return _list.ToList();
    }

    public T? Peek()
    {
        if (_list.Count == 0)
            return default;

        return _list.Last == null ? default : _list.Last.Value;
    }
}