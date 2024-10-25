using Engine.GameApiClient;
using FluentAssertions;

namespace IntegrationTest;

[Explicit]
public class ZorkApiClientTests
{
    [Test]
    public async Task Responsiveness()
    {
        string session = Guid.NewGuid().ToString();
        var sut = new ZorkApiClient();
        var request = new ZorkApiRequest("score", session);
        var response = await sut.GetAsync(request);

        response.Should().NotBeNull();
        response!.Response.Should().NotBeNull();
        Console.WriteLine(response.Response);
    }
}