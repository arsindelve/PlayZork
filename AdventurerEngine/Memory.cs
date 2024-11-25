namespace AdventurerEngine;

/// <summary>
///     Represents a limited-size stack - once you have the maximum number of items, pushing another
///     item deletes the oldest.
/// </summary>
public class Memory(int size)
{
    private List<AdventurerResponse> _list = new();

    public void Push(AdventurerResponse item)
    {
        if (!item.RememberImportance.HasValue)
            return;
        
        _list.Add(item);
        _list = _list.OrderBy(s => s.RememberImportance).ToList();
        if (_list.Count > size) _list.RemoveAt(0);
    }

    public void Degrade()
    {
        _list.ForEach(s => s.RememberImportance *= 0.97m);
    }

    public List<AdventurerResponse> GetAll()
    {
        return _list.ToList();
    }
}