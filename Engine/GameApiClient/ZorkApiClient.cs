using Newtonsoft.Json;
using RestSharp;

namespace Engine.GameApiClient;

public class ZorkApiClient
{
    private static readonly string BaseUrl = "https://bxqzfka0hc.execute-api.us-east-1.amazonaws.com";
    private readonly RestClient _client = new(BaseUrl);

    public async Task<ZorkApiResponse?> GetAsync(ZorkApiRequest resource)
    {
        string requestJson = JsonConvert.SerializeObject(resource);
        var request = new RestRequest("/Prod/ZorkOne", Method.Post); 
        request.AddJsonBody(requestJson); 
        var response = await _client.ExecuteAsync<ZorkApiResponse>(request);
        return response.Data;
    }
}