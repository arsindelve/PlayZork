namespace Engine;

/// <summary>
///     Represents a limited-size stack - once you have the maximum number of items, pushing another
///     item deletes the oldest.
/// </summary>
public class Memory(int size)
{
    private List<GameResponse> _list = new();

    public void Push(GameResponse item)
    {
        _list.Add(item);
        _list = _list.OrderBy(s => s.RememberImportance.GetValueOrDefault()).ToList();
        if (_list.Count > size) _list.RemoveAt(0);
    }

    public List<GameResponse> GetAll()
    {
        return _list.ToList();
    }
}