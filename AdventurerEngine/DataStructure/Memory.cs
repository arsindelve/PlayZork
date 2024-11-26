namespace AdventurerEngine.DataStructure;

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